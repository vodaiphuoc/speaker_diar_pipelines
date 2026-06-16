import math

import torch

from SDP.onnx.diarization.types import SortformerModuleConfig, StreamingSortformerState


class SortformerModules(object):
    r"""
    Copy from original implement of [SortformerModules](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/modules/sortformer_modules.py)
    only `streaming_update_async` method and it dependencies methods.
    Any unnecessary properties, methods are removed
    """

    def __init__(
        self,
        sortformer_config: SortformerModuleConfig,
        sil_threshold: float = 0.2,
        pred_score_threshold: float = 0.25,
        max_index: int = 99999,
        strong_boost_rate: float = 0.75,
        weak_boost_rate: float = 1.5,
        min_pos_scores_rate: float = 0.5,
        scores_boost_latest: float = 0.05,
    ):
        self._sortformer_config = sortformer_config
        self.sil_threshold = sil_threshold
        self.pred_score_threshold = pred_score_threshold
        self.max_index = max_index
        self.strong_boost_rate = strong_boost_rate
        self.weak_boost_rate = weak_boost_rate
        self.min_pos_scores_rate = min_pos_scores_rate
        self.scores_boost_latest = scores_boost_latest

    def streaming_update_async(
        self,
        streaming_state: StreamingSortformerState,
        chunk: torch.Tensor,
        chunk_lengths: torch.Tensor,
        preds: torch.Tensor,
        lc: int = 0,
        rc: int = 0,
    ) -> tuple[StreamingSortformerState, torch.Tensor]:
        """
        Update the speaker cache and FIFO queue with the chunk of embeddings and speaker predictions.
        Asynchronous version, which means speaker cache, FIFO and chunk may have different lengths within a batch.
        Should be used for real streaming applications.

        Args:
            streaming_state (SortformerStreamingState): Previous streaming state including speaker cache and FIFO
            chunk (torch.Tensor): chunk of embeddings to be predicted
                Shape: (batch_size, lc+chunk_len+rc, emb_dim)
            chunk_lengths (torch.Tensor): Lengths of current chunk
                Shape: (batch_size,)
            preds (torch.Tensor): Speaker predictions of the [spkcache + fifo + chunk] embeddings
                Shape: (batch_size, spkcache_len + fifo_len + lc+chunk_len+rc, num_spks)
            lc and rc (int): The left & right offset of the chunk,
                only the chunk[:, lc:chunk_len+lc] is used for update of speaker cache and FIFO queue

        Returns:
            streaming_state (SortformerStreamingState): Current streaming state including speaker cache and FIFO
            chunk_preds (torch.Tensor): Speaker predictions of the chunk embeddings
                Shape: (batch_size, chunk_len, num_spks)
        """
        batch_size, _, emb_dim = chunk.shape
        n_spk = preds.shape[2]

        max_spkcache_len, max_fifo_len, max_chunk_len = (
            streaming_state.spkcache.shape[1],
            streaming_state.fifo.shape[1],
            chunk.shape[1] - lc - rc,
        )

        max_pop_out_len = max(
            self._sortformer_config.spkcache_update_period, max_chunk_len
        )
        max_pop_out_len = min(max_pop_out_len, max_chunk_len + max_fifo_len)

        streaming_state.fifo_preds = torch.zeros(
            (batch_size, max_fifo_len, n_spk), device=preds.device
        )
        chunk_preds = torch.zeros(
            (batch_size, max_chunk_len, n_spk), device=preds.device
        )
        chunk_lengths = (chunk_lengths - lc).clamp(min=0, max=max_chunk_len)
        updated_fifo = torch.zeros(
            (batch_size, max_fifo_len + max_chunk_len, emb_dim), device=preds.device
        )
        updated_fifo_preds = torch.zeros(
            (batch_size, max_fifo_len + max_chunk_len, n_spk), device=preds.device
        )
        updated_spkcache = torch.zeros(
            (batch_size, max_spkcache_len + max_pop_out_len, emb_dim),
            device=preds.device,
        )
        updated_spkcache_preds = torch.full(
            (batch_size, max_spkcache_len + max_pop_out_len, n_spk),
            0.0,
            device=preds.device,
        )

        for batch_index in range(batch_size):
            spkcache_len = streaming_state.spkcache_lengths[batch_index].item()
            fifo_len = streaming_state.fifo_lengths[batch_index].item()
            chunk_len = chunk_lengths[batch_index].item()
            streaming_state.fifo_preds[batch_index, :fifo_len, :] = preds[
                batch_index, spkcache_len : spkcache_len + fifo_len, :
            ]
            chunk_preds[batch_index, :chunk_len, :] = preds[
                batch_index,
                spkcache_len + fifo_len + lc : spkcache_len + fifo_len + lc + chunk_len,
            ]
            updated_spkcache[batch_index, :spkcache_len, :] = streaming_state.spkcache[
                batch_index, :spkcache_len, :
            ]
            updated_spkcache_preds[batch_index, :spkcache_len, :] = (
                streaming_state.spkcache_preds[batch_index, :spkcache_len, :]
            )
            updated_fifo[batch_index, :fifo_len, :] = streaming_state.fifo[
                batch_index, :fifo_len, :
            ]
            updated_fifo_preds[batch_index, :fifo_len, :] = streaming_state.fifo_preds[
                batch_index, :fifo_len, :
            ]

            # append chunk to fifo
            streaming_state.fifo_lengths[batch_index] += chunk_len
            updated_fifo[batch_index, fifo_len : fifo_len + chunk_len, :] = chunk[
                batch_index, lc : lc + chunk_len, :
            ]
            updated_fifo_preds[batch_index, fifo_len : fifo_len + chunk_len, :] = (
                chunk_preds[batch_index, :chunk_len, :]
            )
            if fifo_len + chunk_len > max_fifo_len:
                # move pop_out_len first frames of FIFO queue to speaker cache
                pop_out_len = self._sortformer_config.spkcache_update_period
                pop_out_len = max(pop_out_len, max_chunk_len - max_fifo_len + fifo_len)
                pop_out_len = min(pop_out_len, fifo_len + chunk_len)
                streaming_state.spkcache_lengths[batch_index] += pop_out_len
                pop_out_embs = updated_fifo[batch_index, :pop_out_len, :]
                pop_out_preds = updated_fifo_preds[batch_index, :pop_out_len, :]
                (
                    streaming_state.mean_sil_emb[batch_index : batch_index + 1],
                    streaming_state.n_sil_frames[batch_index : batch_index + 1],
                ) = self._get_silence_profile(
                    streaming_state.mean_sil_emb[batch_index : batch_index + 1],
                    streaming_state.n_sil_frames[batch_index : batch_index + 1],
                    pop_out_embs.unsqueeze(0),
                    pop_out_preds.unsqueeze(0),
                )
                updated_spkcache[
                    batch_index, spkcache_len : spkcache_len + pop_out_len, :
                ] = pop_out_embs
                if updated_spkcache_preds[batch_index, 0, 0] >= 0:
                    # speaker cache already compressed at least once
                    updated_spkcache_preds[
                        batch_index, spkcache_len : spkcache_len + pop_out_len, :
                    ] = pop_out_preds
                elif spkcache_len + pop_out_len > self._sortformer_config.spkcache_len:
                    # will compress speaker cache for the first time
                    updated_spkcache_preds[batch_index, :spkcache_len, :] = preds[
                        batch_index, :spkcache_len, :
                    ]
                    updated_spkcache_preds[
                        batch_index, spkcache_len : spkcache_len + pop_out_len, :
                    ] = pop_out_preds
                streaming_state.fifo_lengths[batch_index] -= pop_out_len
                new_fifo_len = streaming_state.fifo_lengths[batch_index].item()
                updated_fifo[batch_index, :new_fifo_len, :] = updated_fifo[
                    batch_index, pop_out_len : pop_out_len + new_fifo_len, :
                ].clone()
                updated_fifo_preds[batch_index, :new_fifo_len, :] = updated_fifo_preds[
                    batch_index, pop_out_len : pop_out_len + new_fifo_len, :
                ].clone()
                updated_fifo[batch_index, new_fifo_len:, :] = 0
                updated_fifo_preds[batch_index, new_fifo_len:, :] = 0

        streaming_state.fifo = updated_fifo[:, :max_fifo_len, :]
        streaming_state.fifo_preds = updated_fifo_preds[:, :max_fifo_len, :]

        # update speaker cache
        need_compress = (
            streaming_state.spkcache_lengths > self._sortformer_config.spkcache_len
        )
        streaming_state.spkcache = updated_spkcache[
            :, : self._sortformer_config.spkcache_len, :
        ]
        streaming_state.spkcache_preds = updated_spkcache_preds[
            :, : self._sortformer_config.spkcache_len, :
        ]

        idx = torch.where(need_compress)[0]
        if len(idx) > 0:
            streaming_state.spkcache[idx], streaming_state.spkcache_preds[idx], _ = (
                self._compress_spkcache(
                    emb_seq=updated_spkcache[idx],
                    preds=updated_spkcache_preds[idx],
                    mean_sil_emb=streaming_state.mean_sil_emb[idx],
                    permute_spk=False,
                )
            )
            streaming_state.spkcache_lengths[idx] = streaming_state.spkcache_lengths[
                idx
            ].clamp(max=self._sortformer_config.spkcache_len)

        return streaming_state, chunk_preds

    def _boost_topk_scores(
        self,
        scores,
        n_boost_per_spk: int,
        scale_factor: float = 1.0,
        offset: float = 0.5,
    ) -> torch.Tensor:
        """
        Increase `n_boost_per_spk` highest scores for each speaker.

        Args:
            scores (torch.Tensor): Tensor containing scores for each frame and speaker
                Shape: (batch_size, n_frames, n_spk)
            n_boost_per_spk (int): Number of frames to boost per speaker
            scale_factor (float): Scaling factor for boosting scores. Defaults to 1.0.
            offset (float): Offset for score adjustment. Defaults to 0.5.

        Returns:
            scores (torch.Tensor): Tensor containing scores for each frame and speaker after boosting.
                Shape: (batch_size, n_frames, n_spk)
        """
        batch_size, _, n_spk = scores.shape
        _, topk_indices = torch.topk(
            scores, n_boost_per_spk, dim=1, largest=True, sorted=False
        )
        batch_indices = (
            torch.arange(batch_size).unsqueeze(1).unsqueeze(2)
        )  # Shape: (batch_size, 1, 1)
        speaker_indices = (
            torch.arange(n_spk).unsqueeze(0).unsqueeze(0)
        )  # Shape: (1, 1, n_spk)
        # Boost scores corresponding to topk_indices; but scores for disabled frames will remain '-inf'
        scores[batch_indices, topk_indices, speaker_indices] -= scale_factor * math.log(
            offset
        )
        return scores

    def _get_silence_profile(self, mean_sil_emb, n_sil_frames, emb_seq, preds):
        """
        Get updated mean silence embedding and number of silence frames from emb_seq sequence.
        Embeddings are considered as silence if sum of corresponding preds is lower than self.sil_threshold.

        Args:
            mean_sil_emb (torch.Tensor): Previous mean silence embedding tensor
                Shape: (batch_size, emb_dim)
            n_sil_frames (torch.Tensor): Previous number of silence frames
                Shape: (batch_size)
            emb_seq (torch.Tensor): Tensor containing sequence of embeddings
                Shape: (batch_size, n_frames, emb_dim)
            preds (torch.Tensor): Tensor containing speaker activity probabilities
                Shape: (batch_size, n_frames, n_spk)

        Returns:
            mean_sil_emb (torch.Tensor): Updated mean silence embedding tensor
                Shape: (batch_size, emb_dim)
            n_sil_frames (torch.Tensor): Updated number of silence frames
                Shape: (batch_size)
        """
        is_sil = preds.sum(dim=2) < self.sil_threshold
        sil_count = is_sil.sum(dim=1)
        has_new_sil = sil_count > 0
        if not has_new_sil.any():
            return mean_sil_emb, n_sil_frames
        sil_emb_sum = torch.sum(emb_seq * is_sil.unsqueeze(-1), dim=1)
        upd_n_sil_frames = n_sil_frames + sil_count
        old_sil_emb_sum = mean_sil_emb * n_sil_frames.unsqueeze(1)
        total_sil_sum = old_sil_emb_sum + sil_emb_sum
        upd_mean_sil_emb = total_sil_sum / torch.clamp(
            upd_n_sil_frames.unsqueeze(1), min=1
        )
        return upd_mean_sil_emb, upd_n_sil_frames

    def _get_log_pred_scores(self, preds):
        """
        Get per-frame scores for speakers based on their activity probabilities.
        Scores are log-based and designed to be high for confident prediction of non-overlapped speech.

        Args:
            preds (torch.Tensor): Tensor containing speaker activity probabilities
                Shape: (batch_size, n_frames, n_spk)

        Returns:
            scores (torch.Tensor): Tensor containing speaker scores
                Shape: (batch_size, n_frames, n_spk)
        """
        log_probs = torch.log(torch.clamp(preds, min=self.pred_score_threshold))
        log_1_probs = torch.log(torch.clamp(1.0 - preds, min=self.pred_score_threshold))
        log_1_probs_sum = (
            log_1_probs.sum(dim=2)
            .unsqueeze(-1)
            .expand(-1, -1, self._sortformer_config.num_spks)
        )
        scores = log_probs - log_1_probs + log_1_probs_sum - math.log(0.5)
        return scores

    def _get_topk_indices(self, scores):
        """
        Get indices corresponding to spkcache_len highest scores, and binary mask for frames in topk to be disabled.
        Disabled frames correspond to either '-inf' score or spkcache_sil_frames_per_spk frames of extra silence
        Mean silence embedding will be used for these frames.

        Args:
            scores (torch.Tensor): Tensor containing speaker scores, including for extra silence frames
                Shape: (batch_size, n_frames, n_spk)

        Returns:
            topk_indices_sorted (torch.Tensor): Tensor containing frame indices of spkcache_len highest scores
                Shape: (batch_size, spkcache_len)
            is_disabled (torch.Tensor): Tensor containing binary mask for frames in topk to be disabled
                Shape: (batch_size, spkcache_len)
        """
        batch_size, n_frames, _ = scores.shape
        n_frames_no_sil = n_frames - self._sortformer_config.spkcache_sil_frames_per_spk
        # Concatenate scores for all speakers and get spkcache_len frames with highest scores.
        # Replace topk_indices corresponding to '-inf' score with a placeholder index self.max_index.
        scores_flatten = scores.permute(0, 2, 1).reshape(batch_size, -1)
        topk_values, topk_indices = torch.topk(
            scores_flatten, self._sortformer_config.spkcache_len, dim=1, sorted=False
        )
        valid_topk_mask = topk_values != float("-inf")
        topk_indices = torch.where(
            valid_topk_mask, topk_indices, torch.tensor(self.max_index)
        )
        # Sort topk_indices to preserve the original order of the frames.
        # Get correct indices corresponding to the original frames
        topk_indices_sorted, _ = torch.sort(
            topk_indices, dim=1
        )  # Shape: (batch_size, spkcache_len)
        is_disabled = topk_indices_sorted == self.max_index
        topk_indices_sorted = torch.remainder(topk_indices_sorted, n_frames)
        is_disabled += topk_indices_sorted >= n_frames_no_sil
        topk_indices_sorted[is_disabled] = (
            0  # Set a placeholder index to make gather work
        )
        return topk_indices_sorted, is_disabled

    def _gather_spkcache_and_preds(
        self, emb_seq, preds, topk_indices, is_disabled, mean_sil_emb
    ):
        """
        Gather embeddings from emb_seq and speaker activities from preds corresponding to topk_indices.
        For disabled frames, use mean silence embedding and zero probability instead.

        Args:
            emb_seq (torch.Tensor): Tensor containing sequence of embeddings.
                Shape: (batch_size, n_frames, emb_dim)
            preds (torch.Tensor): Tensor containing speaker activity probabilities
                Shape: (batch_size, n_frames, n_spk)
            topk_indices (torch.Tensor): Tensor containing indices of frames to gather
                Shape: (batch_size, spkcache_len)
            is_disabled (torch.Tensor): Tensor containing binary mask for disabled frames
                Shape: (batch_size, spkcache_len)
            mean_sil_emb (torch.Tensor): Tensor containing mean silence embedding
                Shape: (batch_size, emb_dim)

        Returns:
            emb_seq_gathered (torch.Tensor): Tensor containing gathered embeddings.
                Shape: (batch_size, spkcache_len, emb_dim)
            preds_gathered (torch.Tensor): Tensor containing gathered speaker activities.
                Shape: (batch_size, spkcache_len, n_spk)
        """
        # To use `torch.gather`, expand `topk_indices` along the last dimension to match `emb_dim`.
        # Gather the speaker cache embeddings, including the placeholder embeddings for silence frames.
        # Finally, replace the placeholder embeddings with actual mean silence embedding.
        emb_dim, n_spk = emb_seq.shape[2], preds.shape[2]
        indices_expanded_emb = topk_indices.unsqueeze(-1).expand(-1, -1, emb_dim)
        emb_seq_gathered = torch.gather(
            emb_seq, 1, indices_expanded_emb
        )  # (batch_size, spkcache_len, emb_dim)
        mean_sil_emb_expanded = mean_sil_emb.unsqueeze(1).expand(
            -1, self._sortformer_config.spkcache_len, -1
        )
        emb_seq_gathered = torch.where(
            is_disabled.unsqueeze(-1), mean_sil_emb_expanded, emb_seq_gathered
        )

        # To use `torch.gather`, expand `topk_indices` along the last dimension to match `n_spk`.
        # Gather speaker cache predictions `preds`, including the placeholder `preds` for silence frames.
        # Finally, replace the placeholder `preds` with zeros.
        indices_expanded_spk = topk_indices.unsqueeze(-1).expand(-1, -1, n_spk)
        preds_gathered = torch.gather(
            preds, 1, indices_expanded_spk
        )  # (batch_size, spkcache_len, n_spk)
        preds_gathered = torch.where(
            is_disabled.unsqueeze(-1), torch.tensor(0.0), preds_gathered
        )
        return emb_seq_gathered, preds_gathered

    def _get_max_perm_index(self, scores):
        """
        Get number of first speakers having at least one positive score.
        These speakers will be randomly permuted during _compress_spkcache (training only).

        Args:
            scores (torch.Tensor): Tensor containing speaker scores
                Shape: (batch_size, n_frames, n_spk)

        Returns:
            max_perm_index (torch.Tensor): Tensor with number of first speakers to permute
                Shape: (batch_size)
        """

        batch_size, _, n_spk = scores.shape
        is_pos = (
            scores > 0
        )  # positive score usually means that only current speaker is speaking
        zero_indices = torch.where(is_pos.sum(dim=1) == 0)
        max_perm_index = torch.full(
            (batch_size,), n_spk, dtype=torch.long, device=scores.device
        )
        max_perm_index.scatter_reduce_(
            0, zero_indices[0], zero_indices[1], reduce="amin", include_self=False
        )
        return max_perm_index

    def _disable_low_scores(self, preds, scores, min_pos_scores_per_spk: int):
        """
        Sets scores for non-speech to '-inf'.
        Also sets non-positive scores to '-inf', if there are at least min_pos_scores_per_spk positive scores.

        Args:
            preds (torch.Tensor): Tensor containing speaker activity probabilities
                Shape: (batch_size, n_frames, n_spk)
            scores (torch.Tensor): Tensor containing speaker importance scores
                Shape: (batch_size, n_frames, n_spk)
            min_pos_scores_per_spk (int): if number of positive scores for a speaker is greater than this,
                then all non-positive scores for this speaker will be disabled, i.e. set to '-inf'.

        Returns:
            scores (torch.Tensor): Tensor containing speaker scores.
                Shape: (batch_size, n_frames, n_spk)
        """
        # Replace scores for non-speech with '-inf'.
        is_speech = preds > 0.5
        scores = torch.where(is_speech, scores, torch.tensor(float("-inf")))

        # Replace non-positive scores (usually overlapped speech) with '-inf'
        # This will be applied only if a speaker has at least min_pos_scores_per_spk positive-scored frames
        is_pos = (
            scores > 0
        )  # positive score usually means that only current speaker is speaking
        is_nonpos_replace = (
            (~is_pos)
            * is_speech
            * (is_pos.sum(dim=1).unsqueeze(1) >= min_pos_scores_per_spk)
        )
        scores = torch.where(is_nonpos_replace, torch.tensor(float("-inf")), scores)
        return scores

    def _permute_speakers(self, scores, max_perm_index):
        """
        Create a random permutation of scores max_perm_index first speakers.

        Args:
            scores (torch.Tensor): Tensor containing speaker scores
                Shape: (batch_size, n_frames, n_spk)
            max_perm_index (torch.Tensor): Tensor with number of first speakers to permute
                Shape: (batch_size)

        Returns:
            scores (torch.Tensor): Tensor with permuted scores.
                Shape: (batch_size, n_frames, n_spk)
            spk_perm (torch.Tensor): Tensor containing speaker permutation applied to scores
                Shape: (batch_size, n_spk)
        """
        spk_perm_list, scores_list = [], []
        batch_size, _, n_spk = scores.shape
        for batch_index in range(batch_size):
            rand_perm_inds = torch.randperm(max_perm_index[batch_index].item())
            linear_inds = torch.arange(max_perm_index[batch_index].item(), n_spk)
            permutation = torch.cat([rand_perm_inds, linear_inds])
            spk_perm_list.append(permutation)
            scores_list.append(scores[batch_index, :, permutation])
        spk_perm = torch.stack(spk_perm_list).to(scores.device)
        scores = torch.stack(scores_list).to(scores.device)
        return scores, spk_perm

    def _compress_spkcache(
        self, emb_seq, preds, mean_sil_emb, permute_spk: bool = False
    ):
        """
        Compress speaker cache for streaming inference.
        Keep spkcache_len most important frames out of input n_frames, based on preds.

        Args:
            emb_seq (torch.Tensor): Tensor containing n_frames > spkcache_len embeddings
                Shape: (batch_size, n_frames, emb_dim)
            preds (torch.Tensor): Tensor containing n_frames > spkcache_len speaker activity probabilities
                Shape: (batch_size, n_frames, n_spk)
            mean_sil_emb (torch.Tensor): Tensor containing mean silence embedding
                Shape: (batch_size, emb_dim)
            permute_spk (bool): If true, will generate a random permutation of existing speakers

        Returns:
            spkcache (torch.Tensor): Tensor containing spkcache_len most important embeddings from emb_seq.
                Embeddings are ordered by speakers. Within each speaker, original order of frames is kept.
                Shape: (batch_size, spkcache_len, emb_dim)
            spkcache_preds (torch.Tensor): predictions corresponding to speaker cache
                Shape: (batch_size, spkcache_len, n_spk)
            spk_perm (torch.Tensor): random speaker permutation tensor if permute_spk=True, otherwise None
                Shape: (batch_size, n_spk)
        """
        batch_size, n_frames, n_spk = preds.shape
        spkcache_len_per_spk = (
            self._sortformer_config.spkcache_len // n_spk
            - self._sortformer_config.spkcache_sil_frames_per_spk
        )
        strong_boost_per_spk = math.floor(spkcache_len_per_spk * self.strong_boost_rate)
        weak_boost_per_spk = math.floor(spkcache_len_per_spk * self.weak_boost_rate)
        min_pos_scores_per_spk = math.floor(
            spkcache_len_per_spk * self.min_pos_scores_rate
        )

        scores = self._get_log_pred_scores(preds)
        scores = self._disable_low_scores(preds, scores, min_pos_scores_per_spk)

        if permute_spk:  # Generate a random permutation of speakers
            max_perm_index = self._get_max_perm_index(scores)
            scores, spk_perm = self._permute_speakers(scores, max_perm_index)
        else:
            spk_perm = None

        if self.scores_boost_latest > 0:  # Boost newly added frames
            scores[:, self._sortformer_config.spkcache_len :, :] += (
                self.scores_boost_latest
            )

        # Strong boosting to ensure each speaker has at least K frames in speaker cache
        scores = self._boost_topk_scores(scores, strong_boost_per_spk, scale_factor=2)
        # Weak boosting to prevent dominance of one speaker in speaker cache
        scores = self._boost_topk_scores(scores, weak_boost_per_spk, scale_factor=1)

        if (
            self._sortformer_config.spkcache_sil_frames_per_spk > 0
        ):  # Add number of silence frames in the end of each block
            pad = torch.full(
                (
                    batch_size,
                    self._sortformer_config.spkcache_sil_frames_per_spk,
                    n_spk,
                ),
                float("inf"),
                device=scores.device,
            )
            scores = torch.cat(
                [scores, pad], dim=1
            )  # (batch_size, n_frames + spkcache_sil_frames_per_spk, n_spk)

        topk_indices, is_disabled = self._get_topk_indices(scores)
        spkcache, spkcache_preds = self._gather_spkcache_and_preds(
            emb_seq, preds, topk_indices, is_disabled, mean_sil_emb
        )
        return spkcache, spkcache_preds, spk_perm
