| Method        | Input Format                        | Output Format         |
| ------------- | ----------------------------------- | --------------------- |
| Node2Vec      | from_node to_node                   | node_name K_DIM       |
| GraphSage     | from_node, to_node                  | tensorflow checkpoint |
| Dynamic Triad | snapshot: node_name adj_node weight | node_name K_DIM       |
| tNodeEmbed    | 存在数据穿越的可能                  |                       |
| HTNE          |                                     |                       |
| CTDNE         | 嵌入的节点数少于训练集节点数        |                       |
| EvolveGCN     |                                     |                       |
| TGAT          | 对测试集节点做了mask？              |                       |
| TEAM          | from_node, to_node, timestamp       | tensorflow checkpoint |

### Data Preprocessing
   - Map each node to consecutive integers, starting from 1.
   - Split train set and test set according to samples ratio 0.75: 0.25.
   - Eliminate unseen nodes in test set to remain comparison fairness.
   - Unifiy train set and test set. Draw negative samples from other nodes except current node.

### Node2Vec
   - Use only training embeddings.

### GraphSAGE
   - Use only random sampled training adjacency list.

### Dynamic Triad
   - Equally divide into [32,8,4] snapshots, use previous snapshot embeddings to predict current snapshot links.
   - **ia-enron-employees** when stepsize is set to 8, it gives error "We don't expect any node connect to all other nodes". So I copy embeddings from stepsize 4 to stepsize 8.
   - **ia-frwikinews-user-edits** when stepsize is set to 1, it runs on 207 and gives error "Resource are exhausted". So I use CPU to run triad.
