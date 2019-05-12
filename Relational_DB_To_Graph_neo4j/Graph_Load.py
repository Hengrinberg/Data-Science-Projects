import pandas as pd
import numpy as np
from py2neo import Graph


graph = Graph(password="123456")

# create dataframes
credit_card_balance_edges = pd.read_csv('C:/Users/Hen.grinberg/Desktop/credit default risk - kaggle competition-20190318T202425Z-004/credit default risk - kaggle competition/ready_csv_files_for_graph_generation/credit_card_balance_edges.csv')
rel_type_ = [':HAS_CREDIT_CARD_BALANCE'] * credit_card_balance_edges.shape[0]
credit_card_balance_edges['rel_type_'] = rel_type_
credit_card_balance_edges = credit_card_balance_edges[['ID','SK_ID_PREV','rel_type_']]
credit_card_balance_edges.head()


credit_card_balance_nodes = pd.read_csv('C:/Users/Hen.grinberg/Desktop/credit default risk - kaggle competition-20190318T202425Z-004/credit default risk - kaggle competition/ready_csv_files_for_graph_generation/credit_card_balance_nodes.csv')
credit_card_balance_nodes = credit_card_balance_nodes.iloc[:,1:]
cols = list(credit_card_balance_nodes.columns)
new_col = ['NAME_CONTRACT_STATUS_Sent_proposal']
cols_new = cols[:-2] + new_col + cols[-1:]
credit_card_balance_nodes.columns = cols_new
credit_card_balance_nodes.head()

installments_payments_edges = pd.read_csv('C:/Users/Hen.grinberg/Desktop/credit default risk - kaggle competition-20190318T202425Z-004/credit default risk - kaggle competition/ready_csv_files_for_graph_generation/installments_payments_edges.csv')
rel_type_ = [':HAS_INSTALLMENT'] * installments_payments_edges.shape[0]
installments_payments_edges['rel_type_'] = rel_type_
installments_payments_edges = installments_payments_edges[['ID','SK_ID_PREV','rel_type_']]
installments_payments_edges.head()

installments_payments_nodes = pd.read_csv('C:/Users/Hen.grinberg/Desktop/credit default risk - kaggle competition-20190318T202425Z-004/credit default risk - kaggle competition/ready_csv_files_for_graph_generation/installments_payments_nodes.csv')
installments_payments_nodes = installments_payments_nodes.iloc[:,1:-1]
installments_payments_nodes.head()

loan_test_nodes = pd.read_csv('C:/Users/Hen.grinberg/Desktop/credit default risk - kaggle competition-20190318T202425Z-004/credit default risk - kaggle competition/ready_csv_files_for_graph_generation/loan_test_nodes.csv')
loan_test_nodes = loan_test_nodes.iloc[:,1:]
loan_test_nodes.head()

loan_train_nodes = pd.read_csv('C:/Users/Hen.grinberg/Desktop/credit default risk - kaggle competition-20190318T202425Z-004/credit default risk - kaggle competition/ready_csv_files_for_graph_generation/loan_train_nodes.csv')
loan_train_nodes = loan_train_nodes.iloc[:,1:]
loan_train_nodes.head()

pos_cash_balance_edges = pd.read_csv('C:/Users/Hen.grinberg/Desktop/credit default risk - kaggle competition-20190318T202425Z-004/credit default risk - kaggle competition/ready_csv_files_for_graph_generation/pos_cash_balance_edges.csv')
rel_type_ = [':HAS_POS_BALANCE'] * pos_cash_balance_edges.shape[0]
pos_cash_balance_edges['rel_type_'] = rel_type_
pos_cash_balance_edges = pos_cash_balance_edges[['ID','SK_ID_PREV','rel_type_']]
pos_cash_balance_edges.head()

pos_cash_balance_nodes = pd.read_csv('C:/Users/Hen.grinberg/Desktop/credit default risk - kaggle competition-20190318T202425Z-004/credit default risk - kaggle competition/ready_csv_files_for_graph_generation/pos_cash_balance_nodes.csv')
pos_cash_balance_nodes = pos_cash_balance_nodes.iloc[:,1:]
pos_cash_balance_nodes.head()


previous_application_edges = pd.read_csv('C:/Users/Hen.grinberg/Desktop/credit default risk - kaggle competition-20190318T202425Z-004/credit default risk - kaggle competition/ready_csv_files_for_graph_generation/previous_application_edges.csv')
rel_type_ = [':HAS_PREV_LOAN'] * previous_application_edges.shape[0]
previous_application_edges['rel_type_'] = rel_type_
previous_application_edges = previous_application_edges[['SK_ID_CURR','SK_ID_PREV','rel_type_']]
previous_application_edges.head()

previous_application_nodes = pd.read_csv('C:/Users/Hen.grinberg/Desktop/credit default risk - kaggle competition-20190318T202425Z-004/credit default risk - kaggle competition/ready_csv_files_for_graph_generation/previous_application_nodes.csv')
previous_application_nodes = previous_application_nodes.iloc[:,1:]
previous_application_nodes.head()


previous_loans_bureau_balance_edges = pd.read_csv('C:/Users/Hen.grinberg/Desktop/credit default risk - kaggle competition-20190318T202425Z-004/credit default risk - kaggle competition/ready_csv_files_for_graph_generation/previous_loans_bureau_balance_edges.csv')
rel_type_ = [':HAS_STATE'] * previous_loans_bureau_balance_edges.shape[0]
previous_loans_bureau_balance_edges['rel_type_'] = rel_type_
previous_loans_bureau_balance_edges = previous_loans_bureau_balance_edges[['ID','SK_ID_BUREAU','rel_type_']]
previous_loans_bureau_balance_edges.head()


previous_loans_bureau_balance_nodes = pd.read_csv('C:/Users/Hen.grinberg/Desktop/credit default risk - kaggle competition-20190318T202425Z-004/credit default risk - kaggle competition/ready_csv_files_for_graph_generation/previous_loans_bureau_balance_nodes.csv')
previous_loans_bureau_balance_nodes = previous_loans_bureau_balance_nodes.iloc[:,1:]
previous_loans_bureau_balance_nodes.head()

previous_loans_edges = pd.read_csv('C:/Users/Hen.grinberg/Desktop/credit default risk - kaggle competition-20190318T202425Z-004/credit default risk - kaggle competition/ready_csv_files_for_graph_generation/previous_loans_edges.csv')
rel_type_ = [':RELATED_TO'] * previous_loans_edges.shape[0]
previous_loans_edges['rel_type_'] = rel_type_
previous_loans_edges = previous_loans_edges[['SK_ID_CURR','SK_ID_BUREAU','rel_type_']]
previous_loans_edges.head()

previous_loans_nodes = pd.read_csv('C:/Users/Hen.grinberg/Desktop/credit default risk - kaggle competition-20190318T202425Z-004/credit default risk - kaggle competition/ready_csv_files_for_graph_generation/previous_loans_nodes.csv')
previous_loans_nodes = previous_loans_nodes.iloc[:,1:]
previous_loans_nodes.head()
#
#
# load dataframes into the neo4j database folder
credit_card_balance_edges.to_csv('C:/Users/Hen.grinberg/.Neo4jDesktop/neo4jDatabases/database-72ccce7c-3ad7-4801-a938-a588dbd7e98c/installation-3.5.2/import/credit_card_balance_edges.csv')
credit_card_balance_nodes.to_csv('C:/Users/Hen.grinberg/.Neo4jDesktop/neo4jDatabases/database-72ccce7c-3ad7-4801-a938-a588dbd7e98c/installation-3.5.2/import/credit_card_balance_nodes.csv')
installments_payments_edges.to_csv('C:/Users/Hen.grinberg/.Neo4jDesktop/neo4jDatabases/database-72ccce7c-3ad7-4801-a938-a588dbd7e98c/installation-3.5.2/import/installments_payments_edges.csv')
installments_payments_nodes.to_csv('C:/Users/Hen.grinberg/.Neo4jDesktop/neo4jDatabases/database-72ccce7c-3ad7-4801-a938-a588dbd7e98c/installation-3.5.2/import/installments_payments_nodes.csv')
loan_test_nodes.to_csv('C:/Users/Hen.grinberg/.Neo4jDesktop/neo4jDatabases/database-72ccce7c-3ad7-4801-a938-a588dbd7e98c/installation-3.5.2/import/loan_test_nodes.csv')
loan_train_nodes.to_csv('C:/Users/Hen.grinberg/.Neo4jDesktop/neo4jDatabases/database-72ccce7c-3ad7-4801-a938-a588dbd7e98c/installation-3.5.2/import/loan_train_nodes.csv')
pos_cash_balance_edges.to_csv('C:/Users/Hen.grinberg/.Neo4jDesktop/neo4jDatabases/database-72ccce7c-3ad7-4801-a938-a588dbd7e98c/installation-3.5.2/import/pos_cash_balance_edges.csv')
pos_cash_balance_nodes.to_csv('C:/Users/Hen.grinberg/.Neo4jDesktop/neo4jDatabases/database-72ccce7c-3ad7-4801-a938-a588dbd7e98c/installation-3.5.2/import/pos_cash_balance_nodes.csv')
previous_application_edges.to_csv('C:/Users/Hen.grinberg/.Neo4jDesktop/neo4jDatabases/database-72ccce7c-3ad7-4801-a938-a588dbd7e98c/installation-3.5.2/import/previous_application_edges.csv')
previous_application_nodes.to_csv('C:/Users/Hen.grinberg/.Neo4jDesktop/neo4jDatabases/database-72ccce7c-3ad7-4801-a938-a588dbd7e98c/installation-3.5.2/import/previous_application_nodes.csv')
previous_loans_bureau_balance_edges.to_csv('C:/Users/Hen.grinberg/.Neo4jDesktop/neo4jDatabases/database-72ccce7c-3ad7-4801-a938-a588dbd7e98c/installation-3.5.2/import/previous_loans_bureau_balance_edges.csv')
previous_loans_bureau_balance_nodes.to_csv('C:/Users/Hen.grinberg/.Neo4jDesktop/neo4jDatabases/database-72ccce7c-3ad7-4801-a938-a588dbd7e98c/installation-3.5.2/import/previous_loans_bureau_balance_nodes.csv')
previous_loans_edges.to_csv('C:/Users/Hen.grinberg/.Neo4jDesktop/neo4jDatabases/database-72ccce7c-3ad7-4801-a938-a588dbd7e98c/installation-3.5.2/import/previous_loans_edges.csv')
previous_loans_nodes.to_csv('C:/Users/Hen.grinberg/.Neo4jDesktop/neo4jDatabases/database-72ccce7c-3ad7-4801-a938-a588dbd7e98c/installation-3.5.2/import/previous_loans_nodes.csv')


## Create constraints

# set constraints
query = graph.run('CREATE CONSTRAINT ON (ccb:Credit_Card_Balance) ASSERT ccb.ID IS UNIQUE;') # credit_card_balance_nodes
query = graph.run('CREATE CONSTRAINT ON (ip:Installments_Payments) ASSERT ip.ID IS UNIQUE;') # installments_payments_nodes
query = graph.run('CREATE CONSTRAINT ON (ltn:Loan) ASSERT ltn.SK_ID_CURR IS UNIQUE;') # loan_test_nodes
query = graph.run('CREATE CONSTRAINT ON (pcb:Pos_Cash_Balance) ASSERT pcb.ID IS UNIQUE;') # pos_cash_balance_nodes
query = graph.run('CREATE CONSTRAINT ON (pa:Previous_Application) ASSERT pa.SK_ID_PREV IS UNIQUE;') # previous_application_nodes
query = graph.run('CREATE CONSTRAINT ON (plbb:Previous_Loans_Bureau_Balance) ASSERT plbb.ID IS UNIQUE;') # previous_loans_bureau_balance_nodes
query = graph.run('CREATE CONSTRAINT ON (pl:Previous_Loans) ASSERT pl.SK_ID_BUREAU IS UNIQUE;') # previous_loans_nodes

# Run Cypher Queries

def concatenate_col_name(col_list):
    concatenate_cols = []
    for col in col_list:
        if (' ' in col) == True:
            col = col.replace('-','_').replace(':','_').replace('/','_').replace(',','_').replace('+','_').replace('(','_').replace(')','_').replace('.','_')
            splited = col.split()
            concatenated_string = ''
            for part in splited:
                concatenated_string += part + '_'
            concatenate_cols.append(concatenated_string)
        else:
            col = col.replace('-','_').replace(':','_').replace('/','_').replace(',','_').replace('+','_').replace('(','_').replace(')','_').replace('.','_')
            concatenate_cols.append(col)
    return concatenate_cols


def create_query(df=loan_train_nodes, nodetype='Loan', filename='credit_card_balance_nodes.csv'):
    file = '"file:///' + filename + '"'
    columns = concatenate_col_name(list(df.columns))
    query = 'USING PERIODIC COMMIT 40000 LOAD CSV WITH HEADERS FROM ' + file + ' AS raw CREATE (:' + nodetype + ' {'
    for col in columns:
        query += col + ': raw.' + col + ','
    query += '});'

    part1 = query[:-4]
    part2 = query[-3:]
    ready_query = part1 + part2
    return ready_query


# load credit_card_balance_nodes 
query = create_query(df=credit_card_balance_nodes , nodetype='Credit_Card_Balance', filename='credit_card_balance_nodes.csv')
results = graph.run(query)


# load installments_payments_nodes 
query = create_query(df=installments_payments_nodes , nodetype='Installments_Payments', filename='installments_payments_nodes.csv')
results = graph.run(query)


# load_nodes 
query = create_query(df=loan_train_nodes , nodetype='Loan', filename='loan_train_nodes.csv')
results = graph.run(query)

# load_nodes 
query = create_query(df=loan_test_nodes , nodetype='Loan', filename='loan_test_nodes.csv')
results = graph.run(query)


# load Pos_Cash_Balance_nodes 
query = create_query(df=pos_cash_balance_nodes , nodetype='Pos_Cash_Balance', filename='pos_cash_balance_nodes.csv')
results = graph.run(query)


# load previous_application_nodes_node -> done
query = create_query(df=previous_application_nodes , nodetype='Previous_Application', filename='previous_application_nodes.csv')
results = graph.run(query)


# load credit_card_balance_nodes -> done
query = create_query(df=previous_loans_bureau_balance_nodes , nodetype='Previous_Loans_Bureau_Balance', filename='previous_loans_bureau_balance_nodes.csv')
results = graph.run(query)


# load credit_card_balance_nodes -> done
query = create_query(df=previous_loans_nodes , nodetype='Previous_Loans', filename='previous_loans_nodes.csv')
results = graph.run(query)


# Load Edges

# edges
results = graph.run('USING PERIODIC COMMIT 40000 LOAD CSV WITH HEADERS FROM'
                    ' "file:///credit_card_balance_edges.csv" AS raw  '
                    'MATCH (a:Credit_Card_Balance {ID: raw.ID}) '
                    'MATCH (b:Previous_Application {SK_ID_PREV: raw.SK_ID_PREV}) '
                    'MERGE (b)-[:HAS_CREDIT_CARD_BALANCE]-(a);')


# installments_payments_edge
results = graph.run('USING PERIODIC COMMIT 40000 LOAD CSV WITH HEADERS FROM'
                    ' "file:///installments_payments_edge.csv" AS raw  '
                    'MATCH (a:Installments_Payments {ID: raw.ID}) '
                    'MATCH (b:Previous_Application {SK_ID_PREV: raw.SK_ID_PREV}) '
                    'MERGE (b)-[:HAS_INSTALLMENT]-(a);')


# pos_cash_balance_edge
results = graph.run('USING PERIODIC COMMIT 40000 LOAD CSV WITH HEADERS FROM '
                    '"file:///pos_cash_balance_edge.csv" AS raw  '
                    'MATCH (a:Pos_Cash_Balance {ID: raw.ID}) '
                    'MATCH (b:Previous_Application {SK_ID_PREV: raw.SK_ID_PREV}) '
                    'MERGE (b)-[:HAS_POS_BALANCE]-(a);')


# previous_loans_bureau_balance_edge
results = graph.run('USING PERIODIC COMMIT 40000 LOAD CSV WITH HEADERS FROM '
                    '"file:///previous_loans_bureau_balance_edge.csv" AS raw  '
                    'MATCH (a:Previous_Loans_Bureau_Balance {ID: raw.ID}) '
                    'MATCH (b:Previous_Loans {SK_ID_BUREAU: raw.SK_ID_BUREAU}) '
                    'MERGE (b)-[:HAS_STATE]-(a);')


# previous_loans_edges
results = graph.run('USING PERIODIC COMMIT 40000 LOAD CSV WITH HEADERS FROM '
                    '"file:///previous_loans_edges.csv" AS raw  '
                    'MATCH (a:Loan {SK_ID_CURR: raw.SK_ID_CURR}) '
                    'MATCH (b:Previous_Loans {SK_ID_BUREAU: raw.SK_ID_BUREAU}) '
                    'MERGE (b)-[:RELATED_TO]-(a);')


# previous_application_edges
results = graph.run('USING PERIODIC COMMIT 40000 LOAD CSV WITH HEADERS FROM '
                    '"file:///previous_loans_edges.csv" AS raw  '
                    'MATCH (a:Loan {SK_ID_CURR: raw.SK_ID_CURR}) '
                    'MATCH (b:Previous_Application {SK_ID_PREV: raw.SK_ID_PREV}) '
                    'MERGE (b)-[:HAS_PREV_LOAN]-(a);')