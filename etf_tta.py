import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def cosine_logits(proto, etf_vertices, temperature=0.1):
    """
    Cosine similarity between normalized features and frozen ETF vertices.
    """
    proto = F.normalize(proto, dim=-1)
    etf = F.normalize(etf_vertices, dim=-1)
    return (proto @ etf.t()) / temperature


def configure_model_for_etf_tta(model):
    """
    Freeze the entire model except affine parameters of
    BatchNorm and LayerNorm layers.

    BatchNorm:
        - uses batch statistics during TTA
        - does NOT update persistent running statistics
        - affine parameters remain trainable

    LayerNorm:
        - remains in eval mode
        - affine parameters remain trainable
    """
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    adapted = []

    for module in model.modules():

        # ---------------------------------------------------------
        # BatchNorm: use batch statistics, but freeze running stats
        # ---------------------------------------------------------
        if isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
            ),
        ):

            # Use batch statistics during the forward pass,
            # but do not maintain/update persistent running statistics.
            module.train()
            module.track_running_stats = False

            if module.weight is not None:
                module.weight.requires_grad = True
                adapted.append(module.weight)

            if module.bias is not None:
                module.bias.requires_grad = True
                adapted.append(module.bias)

        # ---------------------------------------------------------
        # LayerNorm: no running statistics anyway
        # ---------------------------------------------------------
        elif isinstance(module, nn.LayerNorm):

            module.eval()

            if module.weight is not None:
                module.weight.requires_grad = True
                adapted.append(module.weight)

            if module.bias is not None:
                module.bias.requires_grad = True
                adapted.append(module.bias)

    if not adapted:
        raise RuntimeError(
            "No BatchNorm/LayerNorm affine parameters found for ETF-TTA."
        )

    return adapted

class ETFTTA:
    """
    ETF-aware test-time adaptation.

    Objective:
        L_ETF = - mean_i [ w_i * cosine(z_i, e_hat_i) ]

    where:
        e_hat_i = ETF vertex predicted by the current model
        w_i     = confidence of that prediction

    Only samples with p_max >= confidence_threshold
    participate in the update.

    The ETF itself is always frozen.
    """

    def __init__(
        self,
        model,
        etf_vertices,
        reference_model=None,
        lr=1e-3,
        steps=1,
        temperature=0.1,
        confidence_threshold=0.90,
    ):
        self.model = model
        self.etf_vertices = etf_vertices
        self.reference_model = reference_model
        self.lr = lr
        self.steps = steps
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold

        # ETF is strictly frozen.
        self.etf_vertices.requires_grad_(False)

        # Only normalization affine parameters are adapted.
        self.adapted_params = configure_model_for_etf_tta(self.model)

        self.optimizer = torch.optim.Adam(
            self.adapted_params,
            lr=self.lr,
        )

        self.last_diagnostics = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Adapt on the current batch, then return post-adaptation
        ETF logits.

        Adaptation is non-episodic within a corruption:
        normalization parameters carry forward from batch to batch.
        """

        # ---------------------------------------------------------
        # 1. Source prediction before adaptation
        # ---------------------------------------------------------
        with torch.no_grad():
            proto_before = self.model(x)

            logits_before = cosine_logits(
                proto_before,
                self.etf_vertices,
                temperature=self.temperature,
            )

            probs_before = F.softmax(logits_before, dim=-1)

            # confidence, pseudo_labels = probs_before.max(dim=-1)

            # selected = confidence >= self.confidence_threshold

            confidence, pseudo_labels = probs_before.max(dim=-1)

            # ---------------------------------------------------------
            # Reference-model prediction
            # ---------------------------------------------------------
            if self.reference_model is not None:
                reference_proto = self.reference_model(x)

                reference_logits = cosine_logits(
                    reference_proto,
                    self.etf_vertices,
                    temperature=self.temperature,
                )

                reference_pred = reference_logits.argmax(dim=-1)

                # Only adapt samples whose current prediction agrees
                # with the frozen original model.
                stable = pseudo_labels == reference_pred
            else:
                stable = torch.ones_like(
                    pseudo_labels,
                    dtype=torch.bool,
                )

            # selected = (
            #     (confidence >= self.confidence_threshold)
            #     & stable
            # )
            selected = confidence >= self.confidence_threshold

            source_alignment = (
                F.normalize(proto_before, dim=-1)
                * F.normalize(
                    self.etf_vertices[pseudo_labels],
                    dim=-1,
                )
            ).sum(dim=-1)

        selected_count = int(selected.sum().item())


        stable_fraction = stable.float().mean().item()
        selected_stable_fraction = selected.float().mean().item()

        # ---------------------------------------------------------
        # 2. ETF-aware adaptation
        # ---------------------------------------------------------
        if selected_count > 0:
            for _ in range(self.steps):
                self.optimizer.zero_grad(set_to_none=True)

                proto = self.model(x)

                proto_norm = F.normalize(proto, dim=-1)
                etf_norm = F.normalize(self.etf_vertices, dim=-1)

                cosine_matrix = proto_norm @ etf_norm.t()

                # Pseudo-label is determined from the source
                # prediction and is detached from the graph.
                target_cosine = cosine_matrix[
                    torch.arange(
                        proto.shape[0],
                        device=proto.device,
                    ),
                    pseudo_labels,
                ]

                # Confidence weights are also detached.
                # weights = confidence.detach()
                
                stability_weights = torch.where(
                    stable,
                    torch.ones_like(confidence),
                    torch.full_like(confidence, 0.5), # Original is 0.5
                ).detach()

                weights = confidence.detach() * stability_weights

                # Only sufficiently confident samples adapt.
                weighted_alignment = (
                    target_cosine[selected] * weights[selected]
                )

                loss = -weighted_alignment.mean()

                loss.backward()
                self.optimizer.step()

        # ---------------------------------------------------------
        # 3. Post-adaptation prediction
        # ---------------------------------------------------------
        with torch.no_grad():
            proto_after = self.model(x)

            logits_after = cosine_logits(
                proto_after,
                self.etf_vertices,
                temperature=self.temperature,
            )

            probs_after = F.softmax(logits_after, dim=-1)

            confidence_after, prediction_after = probs_after.max(dim=-1)

            target_after = (
                F.normalize(proto_after, dim=-1)
                * F.normalize(
                    self.etf_vertices[pseudo_labels],
                    dim=-1,
                )
            ).sum(dim=-1)

            target_alignment_before = source_alignment.mean().item()
            target_alignment_after = target_after.mean().item()

            prediction_change_rate = (
                prediction_after != pseudo_labels
            ).float().mean().item()

            entropy_before = (
                -(probs_before * (probs_before + 1e-10).log())
                .sum(dim=-1)
                .mean()
                .item()
            )

            entropy_after = (
                -(probs_after * (probs_after + 1e-10).log())
                .sum(dim=-1)
                .mean()
                .item()
            )



        self.last_diagnostics = {
            "selected_fraction": float(
                selected.float().mean().item()
            ),
            "stable_fraction": float(
                stable_fraction
            ),
            "selected_stable_fraction": float(
                selected_stable_fraction
            ),
            "selected_count": selected_count,
            "alignment_before": float(
                target_alignment_before
            ),
            "alignment_after": float(
                target_alignment_after
            ),
            "prediction_change_rate": float(
                prediction_change_rate
            ),
            "entropy_before": float(
                entropy_before
            ),
            "entropy_after": float(
                entropy_after
            ),
            "confidence_before": float(
                confidence.mean().item()
            ),
            "confidence_after": float(
                confidence_after.mean().item()
            ),
        }

        return logits_after