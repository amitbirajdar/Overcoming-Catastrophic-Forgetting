# Overcoming Catastrophic Forgetting with Hard Attention to the Task

Changes made to the orignial code provided by the author:

1. Network structure changed to make it similar for all approaches in the project
2. Original model trained only on two tasks, MNIST 0-4 and MNIST 5-9. This modified file includes 2 new tasks - permuted MNIST 0-4 and permuted MNIST 5-9
3. Training on all 4 tasks continually

Note: Model works well until 2 tasks but fails in case of 4 sequential tasks. (Could not resolve this issue, any suggesstions appreciated)

## Reference and Link to Paper

Serrà, J., Surís, D., Miron, M. & Karatzoglou, A.. (2018). Overcoming Catastrophic Forgetting with Hard Attention to the Task. Proceedings of the 35th International Conference on Machine Learning, in PMLR 80:4548-4557

Link: [http://proceedings.mlr.press/v80/serra18a.html](http://proceedings.mlr.press/v80/serra18a.html)

Link to github: https://github.com/joansj/hat
