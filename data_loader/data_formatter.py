"""Dynamic graph data from online resources are in different formats. Therefore, it
is needed to be transformed as a unifed graph format.

unifed graph format: Those non-existent features are set as 0.
    dataset.edges: from_node_id, to_node_id, timestamp, state_label, features
    dataset.nodes: node_id, id_map, role, label, features

CTDNE:
    from_node_id, to_node_id, [label?], timestamp
Node Embedding over Temporal Graphs:
    Cora:
        papers: id, filename, author, title, publisher, address, year
        citations: referring_id, cited_id
        citations.withauthors: id, file_name, [cited_id], *, [author], ***
        extractions: authors, title, abstract, etc
    facebook: from_node_id, to_node_id, [**|weight], timestamp
    dblp: xml format, each node is an article: key, author, title, pages, year, volume, journal, url
JODIE:
    data: user_id, item_id, timestamp, state_label, features_list
Neural Graph Collaborative Filtering:
    Yelp:
        business: business_id, name, address, city, state, postal code, latitude, longitude, stars, review_count, is_open, attributes, categories, hours
        review: review_id, user_id, business_id, stars, date, text, useful, funny, cool
        tip: user_id, name
        user: user_id, name, review_count, yelping_since, friends, useful, funny, cool, fans, elite, average_stars, compliment_hot, compliment_more, compliment_profile, compliment_cute, compliment_list,compliment_note,
    Amazon-book review:
        reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
        asin - ID of the product, e.g. 0000013714
        reviewerName - name of the reviewer
        helpful - helpfulness rating of the review, e.g. 2/3
        reviewText - text of the review
        overall - rating of the product
        summary - summary of the review
        unixReviewTime - time of the review (unix time)
        reviewTime - time of the review (raw)
    Gowalla: user_id, check-in time, latitude, longitude, location_id
Aminer:
    # * --- paperTitle
    # @ --- Authors
    # t ---- Year
    # c  --- publication venue
    # index 00---- index id of this paper
    # % ---- the id of references of this paper (there are multiple lines, with each indicating a reference)
    #! --- Abstract
"""
import argparse
import os
import numpy as np
import pandas as pd


def to_csv(project='', data_dir='/nfs/zty/Graph/graph_data/'):
    '''
    Five kinds of datasets from different resources are transformed into a unifed graph format: CTDNE, NEOTG, JODIE, NGCF, Aminer
    '''
    if project not in dirmap.keys():
        raise NotImplementedError(
            "{} dataset is not supported.".format(project))
    project_func, project_name = dirmap[project]
    project_dir = os.path.join(data_dir, project_name)
    project_func(project_dir, '/nfs/zty/Graph/format_data/', project)


def ctdne_transf(project_dir, store_dir, project):
    config = pd.read_csv(f'{project_dir}/README.config')
    bipartite = {row.dataset: row.bipartite for row in config.itertuples()}
    fname = [f for f in os.listdir(project_dir) if f.endswith('.edges')]
    files = [os.path.join(project_dir, f) for f in fname]
    header = ['from_node_id', 'to_node_id', 'timestamp']
    header2 = ['from_node_id', 'to_node_id', 'weight', 'timestamp']
    for f, name in zip(files, fname):
        if name.find('.') != -1:
            name = name[:name.find('.')]
        print('*****{}*****'.format(name))
        if name not in bipartite:
            raise NotImplementedError(name)

        df = pd.read_csv(f, header=None)
        if len(df.columns) < 3:
            # Read non-standard csv files separated by space.
            df = pd.read_csv(f, header=None, sep='\s+')
        if len(df.columns) == 3:
            df.columns = header
        else:
            df.columns = header2

        edges = pd.DataFrame(columns=edges_cols)
        edges[header] = df[header]
        edges['state_label'] = 0

        from_nodes = df['from_node_id'].tolist()
        to_nodes = df['to_node_id'].tolist()
        if bipartite[name]:
            max_from = np.max(from_nodes) + 1
            min_to = np.min(to_nodes)
            to_nodes = np.array(to_nodes) - min_to + max_from
            to_nodes = list(to_nodes)

        edges['to_node_id'] = to_nodes
        edges.to_csv('{}/{}.edges'.format(store_dir, name), index=None)

        nodes_id = sorted(set(from_nodes + to_nodes))
        

        nodes = pd.DataFrame(columns=nodes_cols)
        nodes['node_id'] = nodes_id
        nodes['id_map'] = list(range(len(nodes_id)))
        nodes['role'] = 0
        nodes['label'] = 0
        nodes.to_csv('{}/{}.nodes'.format(store_dir, name), index=None)


def neotg_transf(project_dir, store_dir, project):
    '''
    Cora, DBLP are omitted for simplicity. Citation datasets can be downloaded from Aminer dataset website.
    '''
    fname = ['facebook-wosn-links', 'facebook-wosn-wall', 'slashdot-threads']
    files = ['{0}/out.{0}'.format(f) for f in fname] + ['CollegeMsg.txt']
    files = [project_dir + f for f in files]
    header = ['from_node_id', 'to_node_id', 'timestamp']
    header2 = ['from_node_id', 'to_node_id', 'weight', 'timestamp']
    for f, name in zip(files, fname):
        if name.find('.') != -1:
            name = name[:name.find('.')]
        print('*****{}*****'.format(name))
        df = pd.read_csv(f, header=None, sep=' ')
        if len(df.columns) == 3:
            df.columns = header
        else:
            df.columns = header2

        edges = pd.DataFrame(columns=edges_cols)
        edges[header] = df[header]
        edges['state_label'] = 0
        edges.to_csv('{}/{}-{}.edges'.format(store_dir,
                                             project, name), index=None)

        from_nodes = df['from_node_id'].tolist()
        to_nodes = df['to_node_id'].tolist()
        nodes_id = sorted(set(from_nodes + to_nodes))
        nodes = pd.DataFrame(columns=nodes_cols)
        nodes['node_id'] = nodes_id
        nodes['id_map'] = list(range(1, len(nodes_id) + 1))
        nodes['role'] = 0
        nodes['label'] = 0
        nodes.to_csv('{}/{}-{}.nodes'.format(store_dir,
                                             project, name), index=None)


def jodie_transf(project_dir, store_dir, project):
    fname = [f for f in os.listdir(project_dir)]
    files = [os.path.join(project_dir, f) for f in fname]
    header = ['from_node_id', 'to_node_id', 'timestamp', 'state_label']
    for f, name in zip(files, fname):
        if name.find('.') != -1:
            name = name[:name.find('.')]
        print('*****{}*****'.format(name))
        df = pd.read_csv(f, header=None, skiprows=1)
        headers = header + ['feat{0}'.format(i)
                            for i in range(len(df.columns) - 4)]
        df.columns = headers

        # concat user_id and item_id into a unified id
        # user_id = df['from_node_id'].tolist()
        max_user_id = df.from_node_id.max() + 1
        df["to_node_id"] = df["to_node_id"] + max_user_id
        # item_id = df['to_node_id'].apply(lambda x: x + max_user_id).tolist()
        # df['to_node_id'] = item_id
        df.to_csv('{}/{}-{}.edges'.format(store_dir, project, name), index=None)

        user_id = set(df["from_node_id"])
        item_id = set(df["to_node_id"])
        nodes_id = sorted(user_id.union(item_id))
        print(len(nodes_id), len(user_id), len(item_id))
        nodes = pd.DataFrame(columns=nodes_cols)
        nodes['node_id'] = nodes_id
        nodes['id_map'] = list(range(len(nodes_id)))
        nodes['role'] = ['user'] * len(user_id) + ['item'] * len(item_id)
        nodes['label'] = 0
        nodes.to_csv('{}/{}-{}.nodes'.format(store_dir,
                                             project, name), index=None)


def ngcf_transf(project_dir, store_dir, project):
    pass


def aminer_transf(project_dir, store_dir, project):
    pass


dirmap = {
    'CTDNE': (ctdne_transf, '2018-WWW-CTDNE'),
    'NEOTG': (neotg_transf, '2019-IJCAI-Node-Embedding-over-Temporal-Graphs'),
    'JODIE': (jodie_transf, '2019-KDD-jodie'),
    'NGCF': (ngcf_transf, '2019-SIGIR-Neural-Graph-CF'),
    'Aminer': (aminer_transf, 'Aminer')


}
edges_cols = ['from_node_id', 'to_node_id', 'timestamp', 'state_label']
nodes_cols = ['node_id', 'id_map', 'role', 'label']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, required=True,
                        choices=["CTDNE", "NEOTG", "JODIE"])
    args = parser.parse_args()
    to_csv(args.dataset)
    # to_csv('CTDNE')
    # to_csv('NEOTG')
    # to_csv('JODIE')
