# Dynamic-Graph

## Dataset
In this experiment, we refer to dynamic graph as a graph with temporal interactions between nodes. We do not taKe the increase and decrease of nodes into account for technical reasons. Dynamic graph datasets are downloaded from [networK repository](http://networKrepository.com/ia.php) and [Konect networKs](http://Konect.uni-Koblenz.de/networKs/). Details of the datasets are listed below.

Additionally, we split the datasets into three-fold: train, valid and test datasets with 70:5:25 along the time dimension. We remove unseen nodes in the train dataset from the valid and test datasets. We generate a negative sample for each reserved edge by replacing the `to_node_id` with another random node.

| Dataset                     | Bipartite | Edge Type   | Density | Nodes | Edges  | Test set | Train/Unseen nodes | d_max | d_avg  | Timespan(days) |
| --------------------------- | --------- | ----------- | ------- | ----- | ------ | -------- | ------------------ | ----- | ------ | -------------- |
| ia-workplace-contacts       | False     | Interaction | 2.34    | 92    | 9.8K   | 4.8K     | 91/1               | 1.1K  | 106.8  | 11.43          |
| ia-contacts_hypertext2009   | False     | Proximity   | 3.28    | 113   | 20.8K  | 9.9K     | 111/2              | 1.5K  | 184.2  | 2.46           |
| ia-contact                  | False     | -           | 0.75    | 274   | 28.2K  | 11.2K    | 188/86             | 2.1K  | 103.1  | 3.97           |
| fb-forum                    | False     | -           | 0.08    | 899   | 33.7K  | 15.7K    | 834/65             | 1.8K  | 37.51  | 164.49         |
| soc-sign-bitcoinotc         | False     | -           | 0.002   | 5.8K  | 35.5K  | 6.4K     | 4.4K/1.4K          | 1.2K  | 6.05   | 1903.27        |
| ia-enron-employees          | False     | -           | 4.46    | 151   | 50.5K  | 33.7K    | 148/3              | 5.2K  | 334.9  | 1137.55        |
| ia-escorts-dynamic          | True      | Rating      | 0.0009  | 10K   | 50.6K  | 19.9K    | 6.7K/3.3K          | 616   | 5.01   | 2232.00        |
| ia-reality-call             | False     | Call        | 0.0022  | 7K    | 52.0K  | 1.0K     | 6.7K/86            | 3.0K  | 7.6    | 106.00         |
| ia-retweet-pol              | False     | Retweet     | 0.0003  | 19K   | 61.1K  | 23.0K    | 15K/3.3K           | 1.0K  | 3.3    | 48.78          |
| ia-radoslaw-email           | False     | Email       | 5.98    | 167   | 82.9K  | 41.5K    | 166/1              | 9.1K  | 496.6  | 271.19         |
| ia-movielens-user2tags-10m  | True      | Assignment  | 0.0007  | 17K   | 95.5K  | 33.4K    | 12.7K/3.8K         | 6.0K  | 5.8    | 1108.97        |
| soc-wiki-elec               | False     | -           | 0.004   | 7.1K  | 107.0K | 12.9K    | 5.2K/1.8K          | 1.3K  | 15.04  | 1378.34        |
| ia-primary-school-proximity | False     | Proximity   | 4.31    | 242   | 125.7K | 59.2K    | 242/0              | 2.6K  | 519.7  | 1.35           |
| ia-slashdot-reply-dir       | False     | -           | 0.0001  | 51K   | 140.7K | 27.5K    | 39K/12K            | 3.3K  | 2.76   | 977.36         |
| ia-frwiKinews-user-edits    | False     | Edit        | 0.0006  | 25K   | 193.6K | 58.8K    | 1.9K/5.6K          | 32.6K | 7.73   | 2461.24        |
| JODIE-wikipedia             | True      | -           | 0.0036  | 9.2K  | 157.4K | 59.7K    | 7.4K/1.7K          | 1.9K  | 17.07  | 31.00          |
| JODIE-mooc                  | True      | -           | 0.016   | 7.1K  | 411.7K | 179.0K   | 6.6K/524           | 19.4K | 57.64  | 29.77          |
| JODIE-reddit                | True      | -           | 0.011   | 10.9K | 672.4K | 323.4K   | 10.8K/140          | 58.7K | 61.22  | 31.00          |
| JODIE-lastfm                | True      | -           | 0.66    | 1.9K  | 1.29M  | 457.4K   | 1.7K/188           | 51.7K | 653.08 | 1586.89        |
## Methods
