import torch
from src.hyper_parameters import CAT_OR_ADD


class ReMix_Readout_NoMC(torch.nn.Module):
    """
    Readout with pairwise-causal/spurious-mixing during training.
    """
    def __init__(
            self,
            projection, # Projection 
            mlp # Core MLP classifier (Classification head)
        ):
        super().__init__()
        self.projection = projection
        self.mlp = mlp
    
    def forward(self, h_c, h_s, remix=False):
        """
        The remix flag activates the pairwise-causal/spurious-mixing 
        of h_c and h_s.
            - h_c is the causal hidden representation of the graph.
            - h_s is the spurious hidden representation of the graph.
        We provide comments with the tensor dimensions for our own 
        peace of mind and to make the code easier to follow for the reader.
        """
        if remix:
            # Dim(h_c) = (batch_size, emb_dim_c)
            # Dim(h_s) = (batch_size, emb_dim_s)
            h_c = h_c.unsqueeze(1) # (batch_size, 1         , emb_dim_c)
            h_s = h_s.unsqueeze(0) # (1         , batch_size, emb_dim_s)

            # Pairwise Mixing via broadcasting  
            if CAT_OR_ADD == "cat": # Concatenation (Allows for emb_dim_c != emb_dim_s)
                h_c = h_c.expand(-1, h_s.size(1), -1) # (batch_size, batch_size , emb_dim_c)
                h_s = h_s.expand(h_c.size(0), -1, -1) # (batch_size, batch_size , emb_dim_s)
                h_z = torch.cat([h_c, h_s], dim=-1)   # (batch_size, batch_size, emb_dim),
                                                      #  where emb_dim = emb_dim_c + emb_dim_s
            else: # Addition 
                if not (h_c.size(-1) == h_s.size(-1)): # (Requires emb_dim_c == emb_dim_s)
                    raise ValueError(
                        f"Expected emb_dim_c == emb_dim_s, but got "
                        f"emb_dim_c={h_c.size(-1)}, emb_dim_s={h_s.size(-1)}")
                h_z = h_c + h_s # (batch_size, batch_size, emb_dim)
            
            # Pass via MLP:
            h_z_original_shape = h_z.shape[:-1] # == (batch_size, batch_size, emb_dim)
            h_z = h_z.reshape(-1, h_z.shape[-1]) # (batch_size * batch_size, emb_dim)
            # Core MLP classifier (Classification head)
            logits = self.mlp(self.projection(h_z)) # (batch_size * batch_size, emb_dim)              
            logits = logits.reshape(*h_z_original_shape, -1) # (batch_size, batch_size, num_classes)
            return logits # Logits
            
        else: # No pairwise mixing
            if CAT_OR_ADD == "cat":
                # [causal || spurious]
                h_z = torch.cat([h_c, h_s], dim=-1) 
            else:
                h_z = h_c + h_s # (batch_size, emb_dim)
            # Core MLP classifier (Classification head)
            logits = self.mlp(self.projection(h_z)) # Dim = (batch_size, num_classes)
            return logits # Logits
        