import numpy as np

def gen_matrix(c=0):
    return np.matrix([
        [0,-1,1],
        [1,0,-1],
        [-1-c,1-c,0-c]
    ])


def lemke_howson(A, B, init_label=0):
    """
    Compute one Nash equilibrium of a bimatrix game (A, B) using Lemke-Howson.

    Parameters:
        A (np.array): Payoff matrix for Player 1 (m x n)
        B (np.array): Payoff matrix for Player 2 (m x n)
        init_label (int): Initial label to drop (default 0)

    Returns:
        x (np.array): Mixed strategy for Player 1 (length m)
        y (np.array): Mixed strategy for Player 2 (length n)
    """

    m, n = A.shape

    # Shift payoffs to ensure positivity
    min_payoff = min(np.min(A), np.min(B))
    if min_payoff <= 0:
        A = A - min_payoff + 1
        B = B - min_payoff + 1
    return A

    # Construct the tableau for the linear complementarity problem (LCP)
    M = np.block([
        [np.zeros((n, n)), B.T],
        [A, np.zeros((m, m))]
    ])


    return x, y
if __name__ =="__main__":
    print(lemke_howson(gen_matrix(),np.transpose(gen_matrix())))