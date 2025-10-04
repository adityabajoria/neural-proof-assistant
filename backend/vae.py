import torch
import torch.nn as nn
import torch.nn.functional as F

class ProofVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, latent_dim=20, seq_len=50):
        super(ProofVAE, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Encoder: tokens → embeddings → LSTM → latent
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent → repeated input → LSTM → token logits
        self.decoder_rnn = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        embedded = self.embedding(x)               # (B, T, E)
        _, (h, _) = self.encoder_rnn(embedded)     # h = (1, B, H)
        h = h.squeeze(0)                           # (B, H)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Repeat latent vector across sequence length
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, T, latent_dim)
        out, _ = self.decoder_rnn(z_repeated)                   # (B, T, H)
        logits = self.fc_out(out)                               # (B, T, vocab_size)
        return logits

    def generate(self, z, max_length=100, start_token=1, end_token=2):
        batch_size = z.size(0)
        hidden = None

        # Start with BOS token
        token = torch.tensor([[start_token]], device=z.device)
        outputs = [start_token]

        for _ in range(max_length):
            # Embed the last token
            token_emb = self.embedding(token)          # (B,1,E)

            # Concatenate latent vector (broadcasted) with embedding
            # e.g., project latent_dim → embed_dim first if needed
            latent_in = z.unsqueeze(1)                  # (B,1,L)
            dec_in = latent_in                         # for now, just latent vector
            # Optionally, concat: torch.cat([latent_in, token_emb], dim=-1)

            out, hidden = self.decoder_rnn(dec_in, hidden)   # (B,1,H)
            logits = self.fc_out(out.squeeze(1))             # (B,V)
            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)  # (B,1)
            next_id = next_token.item()
            outputs.append(next_id)

            if next_id == end_token:
                break

            token = next_token  # feed next token in next step

        return outputs


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

    def vae_loss(self, logits, targets, mu, logvar):
        # Reconstruction loss: predict tokens
        recon_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="sum")
        # KL divergence
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld