# this md covers thm 4



the sdp is max sum_ij\notin E (v_ii+v_jj-2V_ij) + sum_ij \in E (-V_ii-V_jj+2V_ij)-nchoose 2 + nd

- in eq 2 to 3, the square is dropped because the big O doesnt care about squared. D^2 is introduced so equation 3 is sort of "normalized"

- both terms in the first part of 3 are positive since in the embedding vi-vj has to be greater than 1 if they arent edges, and less than 1 if they are. 

- in eq 3, the constants 1s are pulled out. the number of edges is E=nd/2, and the number of non edges is E-(n choose 2)/2. 

- doing the sdp below eq 3 makes distances between vertecies with edges 1-epsilon, and distances between non edges as large as possible. 

- **still dont understand the conversion between the primal and dual**
  - zero row addition somehow has to do with translation invariance 

- start w the form A=xI + yJ + zM.
I is identity, J is all ones, M is adj matrix. when a graph is d-regular, all entries look the same in the sense that there are only 2 things that can be said about any point. is a vertex equal to another vertex, and what are the edges/nonedges. (normally you might be able to also say which nodes are more connected)

  - the IJM matricies apparently form a "natural basis"

- **not sure why J is needed**

- we can check our construction of A fits the final condition, there are d 1s in M, the first term and ifnal term cancel, the middle two cancel

- on edges, A_ij = 1-\alpha, on non edges A_ij = 1 and we want edge A_ij to be less than -1 by our optimization problems, thus \alpha\geq 2. 

- finally, we can check that A's eigenvalues are the same as Ms, with an eigenvector u having eigenvalue (\alpha d-n)-\alpha\lambda_i. to ensure this is psd, alpha is set to \frac{n}{d-\lambda_2}. 


- **if this 1/2 trace has to do with the original, then we are done. this circles back to me not understanding the duality conversion**