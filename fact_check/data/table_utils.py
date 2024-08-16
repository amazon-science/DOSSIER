import pandas as pd
import sqlite3
import numpy as np
import tempfile
import multiprocessing
import queue as Queue

def subset_to_sql_schema(tables):
    '''
    Massages tables returned by mimic3_dataset to the schema that SQL queries will run on
    '''
    rename_cols = {
        'adm': {
            'DIAGNOSIS': 'str_label',
            'ADMIT_CUI': 'CUI',
            'rel_t': 't'
        },
        'lab': {
            'value': 'Value',
            'units': 'Units',
            'rel_t': 't'
        },
        'vit': {
            'value': 'Value',
            'units': 'Units',
            'rel_t': 't'
        },
        'input': {
            'AMOUNT': 'Amount',
            'units': 'Units',
            'rel_t': 't'
        },
        'global_kg': {
            'SUBJECT_CUI': 'Subject_CUI',
            'SUBJECT_NAME': 'Subject_Name',
            'PREDICATE': 'Predicate',
            'OBJECT_CUI': 'Object_CUI',
            'OBJECT_NAME': 'Object_Name'
        }
    }

    rename_tables = {
        'adm': 'Admission',
        'vit': 'Vital',
        'input': 'Input',
        'global_kg': 'Global_KG',
        'lab': 'Lab'
    }

    keep_cols = {
        'Admission': ['t', 'str_label', 'CUI'],
        'Vital': ['t', 'CUI', 'Value', 'Units', 'str_label'],
        'Lab': ['t', 'CUI', 'Value', 'Units', 'str_label'],
        'Input': ['t', 'CUI', 'Amount', 'Units', 'str_label'],
        'Global_KG': ['Subject_CUI', 'Subject_Name', 'Object_CUI', 'Object_Name', 'Predicate']
    }

    new_tables = {}

    for tab in tables:
        new_name = rename_tables[tab]
        new_tables[new_name] = (tables[tab].drop(columns = ['ROW_ID', 't'], errors = 'ignore')
                                          .reset_index(drop = True).reset_index().rename(columns = rename_cols[tab])
                                          .rename(columns = {'index': 'ROW_ID'})
                                          [['ROW_ID'] + keep_cols[new_name]])
        if 't' in new_tables[new_name]:
            new_tables[new_name] = new_tables[new_name].sort_values(by = 't')

    return new_tables

def add_identity_edges(df):
    '''
    Adds self loops to global KG
    '''
    sset = set(df['Subject_CUI'].values).union(set(df['Object_CUI'].values))
    new_rows = pd.DataFrame({'Subject_CUI': list(sset),
            'Object_CUI': list(sset),
            'Predicate': 'ISA'})
    new_rows['ROW_ID'] = np.arange(df['ROW_ID'].max()+1, df['ROW_ID'].max()+1 + len(new_rows))
    return pd.concat((df, new_rows), ignore_index = True).drop_duplicates(subset = ['Subject_CUI', 'Object_CUI', 'Predicate'])


def get_sqlite_conn(tables, preprocess = {}):
    '''
    Legacy function. Use make_sqlite_db and add_tables_to_sqlite_db instead.
    '''
    conn = sqlite3.connect(":memory:")
    cur  = conn.cursor()
    for tab in tables:
        if tab in preprocess:
            preprocess[tab](tables[tab]).to_sql(tab, conn, if_exists="replace")
        else:
            tables[tab].to_sql(tab, conn, if_exists="replace")
    cur.execute('CREATE INDEX idx1 ON Global_KG(Object_CUI);') # most searches in Global_KG should be by Object_CUI
    return conn

def open_sqlite_conn(db_path):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return conn

def make_sqlite_db(in_memory = False):
    '''
    Given a dict[str: pd.DataFrame], returns a SQL connection to a SQLite database
    '''
    if in_memory:
        conn = sqlite3.connect(":memory:")
        db_path = None
    else:
        file = tempfile.NamedTemporaryFile(suffix = '.db')
        db_path = file.name
        conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute('PRAGMA journal_mode=wal')
    cur.close()

    return conn, db_path

def add_tables_to_sqlite_db(tables, conn, preprocess = {}, skip = []):    
    for tab in tables:
        if tab in skip:
            continue
        if tab in preprocess:
            preprocess[tab](tables[tab]).to_sql(tab, conn, if_exists="replace")
        else:
            tables[tab].to_sql(tab, conn, if_exists="replace")

        if tab == 'Global_KG':
            cur = conn.cursor()
            cur.execute('CREATE INDEX idx1 ON Global_KG(Object_CUI);') # most searches in Global_KG should be by Object_CUI
            cur.close()

def run_query(query, conn):
    return pd.read_sql_query(query, conn)

def run_query_subprocess(queue, query, db_path):
    '''
    Called by fetch_data_with_timeout; runs a query and put the result in the queue.
    '''
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        df = df.iloc[:1000] # truncate df in case it is huge
        queue.put(df)
    except Exception as e:
        queue.put(e)
    finally:
        if 'conn' in locals():
            conn.close()

def fetch_data_with_timeout(query, db_path, timeout=600):
    '''
    Runs a query against a file-system db in a new process. Times out the process and raises an exception if it doesn't complete.
    '''
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=run_query_subprocess, args=(queue, query, db_path))
    
    process.start()
    try:
        ans = queue.get(timeout=timeout)
    except Queue.Empty:
        process.terminate()
        process.join()
        raise TimeoutError(f"The query took longer than {timeout} seconds to complete.")
    else:
        process.join()

    if isinstance(ans, Exception):
        raise ans
    return ans
