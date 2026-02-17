import numpy as np

def l_method_gatekeeper(
    scores,
    min_keep=1,
    absolute_floor=0.30,
    richness_bias=0.20 # Higher = stricter in high-score batches
):
    """
    RAG gatekeeper using the L-Method (arXiv:2212.12189)
    with a richness bias to enforce strictness in high-signal batches.
    """
    scores = np.sort(np.array(scores))[::-1]
    n = len(scores)
    x = np.arange(1, n + 1)

    if n <= 2: return [True] * n
    
    # 1. Calculate the 'Richness' of the batch
    # If the top score is high, we are in a 'Rich' environment.
    max_s = scores[0]
    is_rich = max_s > 0.85 

    # 2. Find the optimal split point (The L-Method)
    # We find the index 'c' that minimizes the total Root Mean Squared Error 
    # of two lines fitted to the scores.
    best_c = 1
    min_total_rmse = float('inf')

    for c in range(1, n - 1):
        # Line 1: Potential Signal (from 0 to c)
        # Line 2: Potential Noise (from c+1 to n)
        l1_x, l1_y = x[:c+1], scores[:c+1]
        l2_x, l2_y = x[c:], scores[c:]

        # Fit lines and get residuals
        rmse1 = np.std(l1_y - np.polyval(np.polyfit(l1_x, l1_y, 1), l1_x)) if len(l1_y) > 1 else 0
        rmse2 = np.std(l2_y - np.polyval(np.polyfit(l2_x, l2_y, 1), l2_x)) if len(l2_y) > 1 else 0
        
        # Weighted total RMSE
        total_rmse = (c * rmse1 + (n - c) * rmse2) / n
        
        if total_rmse < min_total_rmse:
            min_total_rmse = total_rmse
            best_c = c

    # 3. Apply the 'Inverted' Strictness Requirement
    # If we are in a 'Rich' batch, we treat the 'bend' more aggressively.
    # If the scores are very high, we force a smaller window.
    if is_rich:
        # Strictness Adjustment: In rich batches, we prefer a smaller 'c' 
        # if the signal decay is shallow.
        signal_decay = (scores[0] - scores[best_c]) / max_s
        if signal_decay < richness_bias:
            # If the top documents are all very similar, only take the absolute elite
            best_c = min_keep 

    # 4. Final Thresholding
    threshold = scores[best_c]
    
    # Noise Floor Check: 
    # If the 'relative winner' is below the absolute floor, only keep 1.
    if scores[0] < absolute_floor:
        best_c = min_keep

    mask = [s >= threshold for s in scores]
    return mask

# ===Remark===
# Sometimes off by 1 error present

# --- Testing ---

print("--- SCENARIO A ---")
scores_A = [0.95, 0.80, 0.79, 0.78] 
print(f"Scores: {scores_A}")
print(f"Result: {l_method_gatekeeper(scores_A)}\n")

print("--- SCENARIO B ---")
scores_B = [0.45, 0.30, 0.29, 0.20]
print(f"Scores: {scores_B}")
print(f"Result: {l_method_gatekeeper(scores_B)}\n")

print("--- SCENARIO C ---")
scores_C = [0.99, 0.98, 0.97, 0.96, 0.95]
print(f"Scores: {scores_C}")
print(f"Result: {l_method_gatekeeper(scores_C)}\n")

print("--- SCENARIO D ---")
scores_D = [0.88, 0.87, 0.86, 0.50, 0.49]
print(f"Scores: {scores_D}")
print(f"Result: {l_method_gatekeeper(scores_D)}\n")
