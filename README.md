# Dynamic-Graph

## Dataset
In this experiment, we refer to dynamic graph as a graph with temporal interactions between nodes. We do not taKe the increase and decrease of nodes into account for technical reasons. Dynamic graph datasets are downloaded from [networK repository](http://networKrepository.com/ia.php) and [Konect networKs](http://Konect.uni-Koblenz.de/networKs/). Details of the datasets are listed below.

| Dataset                     | Edge Type   | Density | Nodes | Edges  | Test set | d_max | d_avg | Timespan(days) |
| --------------------------- | ----------- | ------- | ----- | ------ | -------- | ----- | ----- | -------------- |
| ia-workplace-contacts       | Interaction | 2.34    | 92    | 9.8K   | 4.9K     | 1.1K  | 106.8 | 11.43 |
| ia-contacts_hypertext2009   | Proximity   | 3.28    | 113   | 20.8K  | 9.9K     | 1.5K  | 184.2 | 2.46 |
| ia-contact         | -           | 0.75    | 274   | 28.2K  | 13.2K    | 2.1K  | 103.1 | 3.97 |
| fb-forum           | -           | 0.08    | 899   | 33.7K  | -        | 1.8K  | 37.51 | 164.49 | 
| soc-sign-bitcoinotc| -           | 0.002   | 5.8K  | 35.5K  | -        | 1.2K  | 6.05  | 1903.27 |
| ia-enron-employees | -           | 4.46    | 151   | 50.5K  | 25.2K    | 5.2K  | 334.9 | 1137.55 |
| ia-escorts-dynamic | Rating      | 0.0009  | 10K   | 50.6K  | 16.2K    | 616   | 5.01  | 2232.00 |
| ia-reality-call    | Call        | 0.0022  | 7K    | 52.0K  | 20.6K    | 3.0K  | 7.6   | 106.00 |
| ia-retweet-pol     | Retweet     | 0.0003  | 19K   | 61.1K  | 23.7K    | 1.0K  | 3.3   | 48.78 |
| ia-radoslaw-email  | Email       | 5.98    | 167   | 82.9K  | 41.5K    | 9.1K  | 496.6 | 271.19 |
| ia-movielens-user2tags-10m  | Assignment  | 0.0007  | 17K   | 95.5K  | 34.7K    | 6.0K  | 5.8   | 1108.97 |
| soc-wiki-elec               | -           | 0.004   | 7.1K  | 107.0K | -        | 1.3K  | 15.04 | 1378.34 | 
| ia-primary-school-proximity | Proximity   | 4.31    | 242   | 125.7K | 62.9K    | 2.6K  | 519.7 | 1.35 |
| ia-slashdot-reply-dir       | -           | 0.0001  | 51K   | 140.7K | -        | 3.3K  | 2.76  | 13395.39 |
| ia-frwiKinews-user-edits    | Edit        | 0.0006  | 25K   | 193.6K | 61.0K    | 32.6K | 7.73  | 2461.24 |
| JODIE-wikipedia             | -           | 0.0036  | 9.2K  | 157.4K | -        | 1.9K  | 17.07 | 31.00 |
| JODIE-mooc   | - | 0.016 | 7.1K | 411.7K | - | 19.4K | 57.64 | 29.77 |
| JODIE-reddit | - | 0.011 | 10.9K| 672.4K | - | 58.7K | 61.22 | 31.00 |
| JODIE-lastfm | - | 0.66  | 1.9K | 1.29M  | - | 51.7K | 653.08 | 1586.89 | 
## Methods
