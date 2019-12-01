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
    #* --- paperTitle
    #@ --- Authors
    #t ---- Year
    #c  --- publication venue
    #index 00---- index id of this paper
    #% ---- the id of references of this paper (there are multiple lines, with each indicating a reference)
    #! --- Abstract
"""
import os
import pandas as pd


def to_csv(project='', data_dir='/nfs/zty/Graph/graph_data/'):
    '''
    Five kinds of datasets from different resources are transformed into a unifed graph format: CTDNE, NEOTG, JODIE, NGCF, Aminer
    '''
    if project not in dirmap.keys():
        raise NotImplementedError(
            "{} dataset is not supported.".format(project))
    project_func, project_name = dirmap[project]
    project_dir = data_dir + project_name + '/'
    project_func(project_dir, '../graph_data/', project)


def ctdne_transf(project_dir, store_dir, project):
    fname = [f for f in os.listdir(project_dir) if f.endswith('.edges')]
    files = [project_dir + f for f in fname]
    header = ['from_node_id', 'to_node_id', 'timestamp']
    header2 = ['from_node_id', 'to_node_id', 'state_label', 'timestamp']
    for f, name in zip(files, fname):
        if name.find('.') != -1:
            name = name[:name.find('.')]
        print('*****{}*****'.format(name))
        df = pd.read_csv(f, header=None)
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
    files = [project_dir + f for f in fname]
    header = ['from_node_id', 'to_node_id', 'timestamp', 'state_label']
    for f, name in zip(files, fname):
        if name.find('.') != -1:
            name = name[:name.find('.')]
        print('*****{}*****'.format(name))
        df = pd.read_csv(f, header=None, skiprows=1)
        headers = header + ['weight{0}'.format(i)
                            for i in range(len(df.columns) - 4)]
        df.columns = headers

        user_id = df['from_node_id'].tolist()
        max_user_id = max(user_id) * 1000
        item_id = df['to_node_id'].apply(lambda x: x + max_user_id).tolist()
        df['to_node_id'] = item_id
        df.to_csv('{}/{}-{}.edges'.format(store_dir, project, name), index=None)

        nodes_id = sorted(set(user_id + item_id))
        print(len(nodes_id), len(set(user_id)), len(set(item_id)))
        nodes = pd.DataFrame(columns=nodes_cols)
        nodes['node_id'] = nodes_id
        nodes['id_map'] = list(range(1, len(nodes_id) + 1))
        nodes['role'] = ['user'] * \
            len(set(user_id)) + ['item'] * len(set(item_id))
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
    to_csv('CTDNE')
    to_csv('NEOTG')
    # to_csv('JODIE')
