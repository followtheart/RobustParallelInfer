    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights /= value.size(-1) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            attn_weights /= float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention, implement causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            causal_mask_float = causal_mask.float()
            attn_weights = attn_weights * causal_mask_float + (1.0 - causal_mask_float) * mask_value

            #attn_weights = attn_weights * causal_mask + (1.0 - causal_mask) * mask_value

        if attention_mask is not None:
            attn_weights += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.type(value.dtype)  # cast back to V's dtype if necessary
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights *= head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights
