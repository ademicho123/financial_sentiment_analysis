import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_attention(tokens, attention_weights, layer=-1, head=0):
    """
    Visualize the attention weights for a given layer and head.
    
    """
    att_matrix = attention_weights[layer][0, head].detach().cpu().numpy()
    
    # Create a mask to ignore padding tokens
    mask = np.zeros_like(att_matrix)
    for i, token in enumerate(tokens):
        if token == '[PAD]':
            mask[i, :] = 1
            mask[:, i] = 1
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(att_matrix, 
                xticklabels=tokens, 
                yticklabels=tokens, 
                cmap='YlOrRd', 
                mask=mask)
    plt.title(f'Attention weights (Layer {layer}, Head {head})')
    plt.tight_layout()
    return plt