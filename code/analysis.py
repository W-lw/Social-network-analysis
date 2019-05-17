import sqlite3
import pandas as pd
import collections
import networkx as nx
from collections import defaultdict
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl

import warnings
warnings.filterwarnings("ignore")

#-------------用户样本特征---------------
def plot_user_distribution():
    '''
    对所有用户的特征进行统计
    '''
    conn = sqlite3.connect("zhihu.db")
    # 将数据库中的数据加载到conn变量中
    # 然后将User表中的数据都传递给user_data,数据格式是pandas.DataFrame 
    user_data = pd.read_sql('select * from User', conn)    
    conn.close()
    feature_list = ['followee_num', 'follower_num', 'answer_num', 'agree_num', 'thanks_num']   
    '''
    关注数,粉丝数,回答数,赞同数,感谢数
    '''
    
    for feature in feature_list:    
        pl.figure(feature)
        pl.title(feature)
        pl.xlabel("users(sum=26161)")
        pl.ylabel("number of "+feature)
        user = list(range(len(user_data)))

        #对用户特征的数目进行排序
        feature_value = sorted(list(user_data[feature]), reverse=True)
        pl.scatter(user, feature_value)
        print("所有用户{feature}特征的平均数是:".format(feature = feature),np.mean(list(user_data[feature])))
        print("所有用户{feature}特征的中位数是:".format(feature = feature),np.median(list(user_data[feature])))
        print("所有用户{feature}特征的标准差是:".format(feature = feature),np.std(list(user_data[feature])))
        # print('mean of', feature, np.mean(list(user_data[feature])))
        # print('median of', feature, np.median(list(user_data[feature])))
        # print('standard deviation of', np.std(list(user_data[feature])), '\n')
        pl.savefig('./png/所有用户的{feature}特征分布'.format(feature = feature))

        # pylab.show()
    

def plot_user_summary_log_log_distribution():
    '''
    对用户的不同特征对应的数目/粉丝数进行对数回归统计，相当于统计4种特征网络中出度与入度的分布情况
    '''
    conn = sqlite3.connect("zhihu.db")
    user_data = pd.read_sql('select * from User where answer_num >10 and (agree_num > 100 and thanks_num > 100) ', conn)    
    conn.close()
    
    feature_list = ['follower_num']#, 'follower_num', 'answer_num', 'agree_num', 'thanks_num']   
    
    for feature in feature_list:    
        pl.figure('log-log' + feature)
        pl.title('Log-log Distribution of ' + feature + ' to User Count')
        pl.xlabel(feature + ' Count(log10)')
        pl.ylabel("User Count(log10)")
        
        feature_count_pairs = collections.Counter(list(user_data[feature])).most_common()
        feature_value = np.log10(list(zip(*feature_count_pairs))[0])
        user_count = np.log10(list(zip(*feature_count_pairs))[1])
        pl.scatter(feature_value, user_count,color = 'deeppink')
#        pl.savefig('./png/所有用户{feature}对粉丝数的出入度分布情况'.format(feature= feature))

        pl.show()
    

def plot_user_agree_and_follower_correlation():
    '''
    用户赞同数和用户粉丝数的相关性统计
    '''
    conn = sqlite3.connect("zhihu.db")
    user_data = pd.read_sql('select * from User', conn) #database data -> pandas.DataFrame    
    conn.close()
    
    pl.figure('agree and follower')    
    pl.title('Correlation Between Agree Count and Follower Count')
    pl.xlabel('Follower Count(log10)')
    pl.ylabel("Thanks Count(log10)")   
    agree_num = np.log10(list(user_data['thanks_num']))
    follower_num = np.log10(list(user_data['follower_num']))
    pl.scatter(follower_num, agree_num)
    
    # pylab.show()
    pl.savefig('./png/用户赞同数与粉丝数的相关性统计图')

def plot_followee_num_and_follower_correlation():
    '''
    用户赞同数和用户粉丝数的相关性统计
    '''
    conn = sqlite3.connect("zhihu.db")
    user_data = pd.read_sql('select * from User ', conn) #database data -> pandas.DataFrame    
    conn.close()
    
    pl.figure('follower and follower')    
    pl.title('Correlation Between follower Count and answer Count')
    pl.xlabel('Answer Count(log10)')
    pl.ylabel("Thanks/Answer Count(log10)")   
    y_ = np.log10(list(user_data['follower_num']))
    
    y_ =[]
    agree_num = list(user_data['agree_num'])
    thanks_num = list(user_data['thanks_num'])
    answer_num = list(user_data['answer_num'])
    for t in range(len(answer_num)):
        y_.append(thanks_num[t]/(answer_num[t]+1))
    
    x_ = np.log10(list(user_data['answer_num']))
#    followee_num = list(user_data['followee_num'])
#    follower_num = list(user_data['follower_num'])
    
    pl.scatter(x_, y_,color = 'y')
#    x = range(7)
#    y = x
#    pl.plot(x,y,color = 'b')
#    pl.show()
#    pl.savefig('./png/用户赞同数与粉丝数的相关性统计图')

#--------------网络特征-------------

def density_centrality():
    '''
    对网络中图的密度和点的中心度进行统计
    '''
    conn = sqlite3.connect("zhihu.db")   

    #net50K  
    following_data = pd.read_sql('select user_url, followee_url from Following where followee_url in (select user_url from User where agree_num > 50000) and user_url in (select user_url from User where agree_num > 50000)', conn)        
    #net10K
    #following_data = pd.read_sql('select user_url, followee_url from Following where followee_url in (select user_url from User where agree_num > 10000) and user_url in (select user_url from User where agree_num > 10000)', conn)        
    conn.close()

    G = nx.DiGraph()
    for d in following_data.iterrows():
        G.add_edge(d[1][0], d[1][1])

    #图的平均最短路径长度:
    print('图的平均最短路径长度:',nx.average_shortest_path_length(G))

    #图的密度
    print('图的密度:', nx.density(G))

    #python2
    # user_betweenness_list = sorted(nx.betweenness_centrality(G).items(), lambda x, y: cmp(x[1], y[1]), reverse=True) #result like [(2, 0.0), (3, 0.0), (1, 1.0)]
    #python3.x
    user_betweenness_list = sorted(nx.betweenness_centrality(G).items(), key = lambda x:x[1], reverse=True)
    
    betweenness_list = list(zip(*user_betweenness_list))[1]#[(2, 3, 1), (0.0, 0.0, 1.0)][1]

    pl.figure('Betweenness Distribution')
    pl.title('Betweenness Distribution')
    pl.xlabel('Indivisual User')
    pl.ylabel('Betweeness of the User')       
    pl.scatter(list(range(len(betweenness_list))), betweenness_list)
    pl.savefig('./png/点的介性中心度Betweenness Centrality统计')

    #python2
    # user_closeness_list = sorted(nx.closeness_centrality(G).items(), lambda x, y: cmp(x[1], y[1]), reverse=True) #Dict.items(): {1: 1.0, 2: 0.0, 3: 0.0} -> [(1, 1.0), (2, 0.0), (3, 0.0)]
    #python3.x
    user_closeness_list = sorted(nx.closeness_centrality(G).items(), key = lambda x : x[1] , reverse=True) # {1: 1.0, 2: 0.0, 3: 0.0} -> [(1, 1.0), (2, 0.0), (3, 0.0)]
    
    closeness_list = list(zip(*user_closeness_list))[1]   

    pl.figure('Closeness Distribution')
    pl.title('Closeness Distribution')
    pl.xlabel('Indivisual User')
    pl.ylabel('Closeness of the User')       
    pl.scatter(list(range(len(closeness_list))), closeness_list)
    pl.savefig('./png/点的近性中心度Closeness Centrality统计')


def strongly_connected_components():
    '''
    图的强连通分量
    '''
    conn = sqlite3.connect("zhihu.db") 

    #net50K   
    following_data = pd.read_sql('select user_url, followee_url from Following where followee_url in (select user_url from User where follower_num < 100) and user_url in (select user_url from User where follower_num < 100)', conn)        
    #net10K
#    following_data = pd.read_sql('select user_url, followee_url from Following where followee_url in (select user_url from User where agree_num > 10000) and user_url in (select user_url from User where agree_num > 10000)', conn)        
    conn.close()
    
    G = nx.DiGraph()
    cnt = 0
    for d in following_data.iterrows():
        G.add_edge(d[1][0],d[1][1])
        cnt += 1
    print('网络中链的数量:', cnt)

    scompgraphs = nx.strongly_connected_component_subgraphs(G)
    scomponents = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
    print('强连通分量中节点个数', [len(c) for c in scomponents])
    
    #强连通分量中平均最短路径长度(>=1 nodes)
    index = 0
    print('强连通分量中平均最短路径长度(>=1 nodes)')
    for tempg in scompgraphs:
        index += 1
        if len(tempg.nodes()) != 1:
            print(nx.average_shortest_path_length(tempg))
            print('直径', nx.diameter(tempg))
            print('半径', nx.radius(tempg))
#            nx.draw_kamada_kawai(tempg)
#            nx.draw_kamada_kawai(tempg,node_color = 'red',node_shape = '8',alpha=0.8,edge_color='aqua')
#            pl.show()
#            pl.savefig('./png/bigcom60k')
            nx.draw_spring(tempg,node_color = 'red',node_shape = '8',alpha=0.8,edge_color='dimgrey')
            pl.show()
#            pl.figure(index)
            pl.savefig('./png/Biggest_component100')
        # pylab.show()
        # pl.savefig('第{i}个强连通分量图'.format(i =index))

    # Components-as-nodes Graph
    cG = nx.condensation(G)
    pl.figure('Components-as-nodes Graph')
    nx.draw_spring(cG,node_color = 'r',node_shape = '8',alpha=0.8,edge_color='fuchsia')
    # pylab.show()  
    pl.savefig('./png/Components-as-nodes Graph')  

def dominant_set_topic_rank():
    '''
    热门话题分析
    '''
    conn = sqlite3.connect("zhihu.db")     
    following_data = pd.read_sql('select user_url, followee_url from Following where followee_url in (select user_url from User where agree_num > 30000) and user_url in (select user_url from User where agree_num > 30000)', conn)        
    G = nx.DiGraph()
    for d in following_data.iterrows():
        G.add_edge(d[1][0], d[1][1])
    
    #获得支配集
    dominant_set = nx.dominating_set(G)
    print('支配集中的用户数量:', len(dominant_set))

    #支配集中用户回答的话题
    user_topic_data = pd.read_sql('select user_url, topic from UserTopic', conn) 
       
    topicdict = defaultdict(int)
    i = 0   #counter
    for row in user_topic_data.iterrows():
        user_url = row[1][0]
        topic = row[1][1]
        if user_url in dominant_set:
            topicdict[topic] += 1
        i += 1
        #if i % 100 == 0:
            #print i
    conn.close()
    
    topicsorted = sorted(topicdict.items(), key=lambda x: x[1], reverse=True)
    
    #前10的话题
    for t in topicsorted[:10]:
        print(t[0],t[1])

#------------用户影响力发掘---------------

def pagerank_hits():
    conn = sqlite3.connect("zhihu.db")     
    #following_data = pd.read_sql('select user_url, followee_url from Following where followee_url in (select user_url from User where agree_num > 50000) and user_url in (select user_url from User where agree_num > 50000)', conn)        
    following_data = pd.read_sql('select user_url, followee_url from Following where followee_url in (select user_url from User where agree_num > 150000) and user_url in (select user_url from User where agree_num > 600000)', conn)        
    conn.close()

    
    G = nx.DiGraph()
    cnt = 0
    for d in following_data.iterrows():
        G.add_edge(d[1][0],d[1][1])
        cnt += 1
    print('网络中边的数量', cnt)
    pl.figure(0)
    nx.draw_networkx(G)
    # pl.show()
    pl.savefig('整个网络图')

    # PageRank
    pr = nx.pagerank(G)
    prsorted = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    print('基于pagerank算法的用户影响力前10名\n')
    for p in prsorted[:10]:
        print(p[0], p[1])
    
    # HITS
    hub, auth = nx.hits(G)
    print('基于HITS(hub)算法的用户影响力前10名\n')
    for h in sorted(hub.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(h[0], h[1])
    print('基于HITS(auth)算法的用户影响力前10名\n')    
    for a in sorted(auth.items(), key=lambda x: x[1], reverse=True)[:10]:     
        print(a[0], a[1])



if __name__ == '__main__':
    
    if not os.path.exists('./png/'):
        os.makedirs('./png/')

    '''
    用户统计
    '''
#    plot_user_summary_log_log_distribution()
#    plot_user_agree_and_follower_correlation()
    plot_followee_num_and_follower_correlation()
#    strongly_connected_components()
    # plot_user_distribution()
#    plot_followee_num_and_follower_correlation()
    # plot_user_agree_and_follower_correlation()

    '''
    网络结构
    '''
    
    #density_centrality()
#    strongly_connected_components()
# dominant_set_topic_rank()

# pagerank_hits()
