
import pandas as pd

# Read the Excel file
customer_path = 'C:/Users/tdhamodaran/flask_app/data/customers_data_export.xlsx'
product_path = 'C:/Users/tdhamodaran/flask_app/data/products_data_export.xlsx'
spares_path = 'C:/Users/tdhamodaran/flask_app/data/spares_data_export.xlsx'
service_data_path = 'C:/Users/tdhamodaran/flask_app/data/service_data_data_export.xlsx'
employee_path = 'C:/Users/tdhamodaran/flask_app/data/employees_data_export.xlsx'

df_customer = pd.read_excel(customer_path, sheet_name='Sheet1')  # Adjust sheet_name as needed
df_product = pd.read_excel(product_path, sheet_name='Sheet1')  # Adjust sheet_name as needed
df_spares = pd.read_excel(spares_path, sheet_name='Sheet1')  # Adjust sheet_name as needed
df_service_data = pd.read_excel(service_data_path, sheet_name='Sheet1')  # Adjust sheet_name as needed
df_employee = pd.read_excel(employee_path, sheet_name='Sheet1')  # Adjust sheet_name as needed


import sqlite3

# Connect to SQLite database (or create it if it doesn't exist)
sqlite_db_path = ('C:/Users/tdhamodaran/sqliteDB/Retex.db')
conn = sqlite3.connect(sqlite_db_path)
cursor = conn.cursor()

drop_table_queries = ['drop table if exists customers',
'drop table if exists spares',
'drop table if exists products',
'drop table if exists employees',
'drop table if exists service_data']

# Execute each drop table query
for query in drop_table_queries:
    cursor.execute(query)
    conn.commit()

create_table_queries = [
  '''
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_name TEXT NOT NULL,
        customer_since TEXT NOT NULL
    )
    ''',
    '''
    CREATE TABLE employees (
        employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        Designation TEXT
    )
    ''',
    '''
    CREATE TABLE products (
        product_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_name TEXT NOT NULL,
        model_number TEXT NOT NULL
    )
    ''',
    '''
    CREATE TABLE spares (
        spare_id INTEGER PRIMARY KEY AUTOINCREMENT,
        spare_name TEXT
    )
    ''',
    '''
    CREATE TABLE service_data (
        service_data_record_id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_product_id INTEGER NOT NULL,
        issue_date DATE,
        status TEXT,
        spares_used TEXT,
        is_warranty TEXT,
        assigned_to INTEGER,
        FOREIGN KEY (customer_product_id) REFERENCES customer_product (customer_product_id),
        FOREIGN KEY (assigned_to) REFERENCES employees (employee_id)
    )
    '''
]
# Execute each table creation query separately
for query in create_table_queries:
    cursor.execute(query)
    conn.commit()


# Define DataFrames and corresponding table names
dataframes = [df_customer, df_spares, df_product,df_service_data,df_employee]
table_names = ['customers', 'spares', 'products','service_data','employees']

# Insert data into the SQLite tables
for df, table_name in zip(dataframes, table_names):
    df.to_sql(table_name, conn, if_exists='append', index=False)

# Close the connection
conn.close()
