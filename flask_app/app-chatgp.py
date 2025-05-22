from flask import Flask, render_template, request, redirect, url_for, send_file,flash,jsonify
from flask_mail import Mail, Message
from werkzeug.exceptions import abort
import sqlite3
import pandas as pd
import io
import plotly.express as px
import secrets
import hashlib
import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import openai
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from chatbot import get_chatbot_response
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

# Initialize OpenAI API#
openai.api_key = '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'


# Configuration for Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Replace with your SMTP server
app.config['MAIL_PORT'] = 587  # Common port for SMTP
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'thilipdhamodaran@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'cuxn quqw amoe xwrq'
app.config['MAIL_DEFAULT_SENDER'] = 'thilipdhamodaran@gmail.com'
mail = Mail(app)


# Generate a random string and hash it
random_string = os.urandom(32)
hash_value = hashlib.sha256(random_string).hexdigest()

# Set the hash value as the secret key
app.secret_key = hash_value

with open('C:/Users/vijig/Documents/GitHub/CRM-Data-Management/flask_app/key/credentials.json') as json_file:
    credentials_info = json.load(json_file)
    credentials = service_account.Credentials.from_service_account_info(
    credentials_info,
    scopes=['https://www.googleapis.com/auth/drive.file']
)
    

class ActionGenerateSQLQuery(Action):
    def name(self):
        return "action_generate_sql_query"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict):
        
        # Get the latest user message
        user_message = tracker.latest_message.get('text')

  

        # Connect to SQLite database
        conn = sqlite3.connect('your_database.db')
        cursor = conn.cursor()

        try:
            # Execute the generated query
            cursor.execute(generated_query)
            results = cursor.fetchall()

            # Format the response
            response = "Here are the results:\n"
            for row in results:
                response += f"{row}\n"

            dispatcher.utter_message(text=response)

        except sqlite3.Error as e:
            dispatcher.utter_message(text=f"An error occurred: {e}")

        conn.close()
        return []
    


    
# Use OpenAI API to generate SQL query
def generate_sql_from_natural_language(natural_language_query):
        response = openai.Completion.create(
            engine="gpt-4o-mini",
            prompt=f"I have product details in products table and customer details in customers table. Generate SQL Query based on user input: {natural_language_query}",
            max_tokens=150
        )
        return response.choices[0].text.strip()

# Function to connect to the SQLite database
def get_db_connection():
    conn = sqlite3.connect('C:/Users/vijig/Documents/GitHub/CRM-Data-Management/Retex.db')
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL;')
    return conn

def query_database(sql_query):
    conn = get_db_connection()
    cursor = conn.cursor()
    logging.debug(f"executing:{sql_query}")
    cursor.execute(sql_query)
    results = cursor.fetchall()
    conn.close()
    return results

def get_data_for_service_graph():
    # Connect to the SQLite database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Execute a query to retrieve data
    cursor.execute("""
    SELECT  status,strftime('%m', issue_date) AS issue_month, COUNT(status) AS status_count 
     FROM  service_data WHERE strftime('%Y', issue_date) = '2024' AND status <> 'Closed'
    GROUP BY status,strftime('%m', issue_date);""")
    
    # Fetch all results as a list of tuples
    data = cursor.fetchall()
    # Close the connection
    conn.close()
    return data


def create_service_graph(data):
    # Convert the list of tuples into a DataFrame
    df = pd.DataFrame(data, columns=['status', 'issue_month', 'status_count'])
    # Create a bar chart
    fig = px.bar(df, x='issue_month', y='status_count', color='status', title='Open Service calls per Month')
    # Return the HTML representation of the graph
    return fig.to_html(full_html=False)

def get_data_for_products_table():
    # Connect to the SQLite database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Execute a query to retrieve data
    cursor.execute(""" SELECT p.product_name || '-' || p.model_number AS product_info, 
        COUNT(*) AS product_count FROM service_data s 
    INNER JOIN customer_product cp on cp.customer_product_id=s.customer_product_id
    inner join products p ON cp.product_id = p.product_id
    GROUP BY product_info, p.model_number
    ORDER BY product_count DESC LIMIT 10;
    """)
    
    # Fetch all results as a list of tuples
    data = cursor.fetchall()
    # Close the connection
    conn.close()
    return data

def fetch_data(query):
    conn = get_db_connection()
    cursor = conn.execute(query)
    data = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return data

def get_data_for_top_products_table():
    # Connect to the SQLite database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Execute a query to retrieve data
    cursor.execute(""" SELECT p.product_name || '-' || p.model_number  as product_info,count(cp.customer_id) as Customers_using 
from products p inner join customer_product cp on p.product_id=cp.product_id
                   order by Customers_using desc limit 10;
    """)
    
    # Fetch all results as a list of tuples
    data = cursor.fetchall()
    # Close the connection
    conn.close()
    return data

def get_top_customers():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = """
       SELECT c.customer_id, c.customer_name, COUNT(cp.product_id) AS product_count,
               GROUP_CONCAT(p.product_name, ', ') AS product_names
        FROM customers c
        JOIN customer_product cp ON c.customer_id = cp.customer_id
        JOIN products p ON cp.product_id = p.product_id
        GROUP BY c.customer_id, c.customer_name
        ORDER BY product_count DESC
        LIMIT 10;
        """
        cursor.execute(query)
        top_customers = cursor.fetchall()
    except Exception as e:
        flash(f'An error occurred while fetching top customers: {str(e)}', 'error')
        top_customers = []
    finally:
        conn.close()
    return top_customers

def get_open_calls():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = """
       select c.customer_name,p.product_name,p.model_number,
    sd.issue_date,sd.status,sd.is_warranty from service_data sd  inner join customer_product cp on cp.customer_product_id=sd.customer_product_id  
    inner join customers c on c.customer_id=cp.customer_id inner join
    products p on p.product_id=cp.product_id inner join
    employees e on e.employee_id=sd.assigned_to
    where status<>'Closed'
        """
        cursor.execute(query)
        open_calls = cursor.fetchall()
    except Exception as e:
        flash(f'An error occurred while fetching open_calls: {str(e)}', 'error')
        open_calls = []
    finally:
        conn.close()
    return open_calls

# Route for pages


@app.route('/')
def home():
    data = get_data_for_service_graph()
    graph = create_service_graph(data)
    product_service_data = get_data_for_products_table()
    top_product_data = get_data_for_top_products_table()
    top_customers = get_top_customers()  # Fetch top customers data
    open_Calls = get_open_calls()  # Fetch top customers data
    # product_graph = create_product_graph(product_data)
    return render_template('home.html' , graph=graph,product_service_data=product_service_data,top_product_data=top_product_data,top_customers=top_customers,open_Calls=open_Calls)

@app.route('/chat', methods=['POST'])
def chat():
    user_input= request.json.get('query')
    sql_query = generate_sql_from_natural_language(user_input)
    if sql_query:
        results=query_database(sql_query)
        return jsonify(results)
    else:
        return jsonify({"error":"Could not understand the query"}),400

#Products
@app.route('/products', methods=['GET'])
def products():
    search_query = request.args.get('search', '')
    show_all = request.args.get('show_all', 'false').lower() == 'true'
    conn = get_db_connection()
    
    if conn is None:
        flash('Database connection failed!', 'error')
        return render_template('error_page.html')  # Create an error page template

    cursor = conn.cursor()
    products = []
    try:
        if show_all:
            query = "SELECT product_id, product_name, model_number FROM products"
            products = conn.execute(query).fetchall()
        elif search_query:
            query = "SELECT product_id, product_name, model_number FROM products WHERE product_name LIKE ?"
            products = conn.execute(query, ('%' + search_query + '%',)).fetchall()
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
    finally:
        conn.close()

    return render_template('products/products.html', products=products, search=search_query, show_all=show_all)

@app.route('/add_product', methods=('GET', 'POST'))
def add_product():
    if request.method == 'POST':
        product_name = request.form['product_name'] 
        model_number = request.form['model_number'] 
        if not product_name or not model_number:
            flash('Product Name and Model Number are required!', 'error')
            return redirect(url_for('products'))
        conn = get_db_connection()
        conn.execute('INSERT INTO products (product_name,model_number) VALUES (?,?)', (product_name,model_number))
        conn.commit()
        conn.close()
    
        flash('New product added successfully!', 'success')
        
    return render_template('products/add_product.html')

@app.route('/product/<int:product_id>')
def product_details(product_id):
    conn = get_db_connection()
    # Query 1: Fetch product details
    product_query = "SELECT product_id,product_name, model_number FROM products WHERE product_id = ?"
    product_data = conn.execute(product_query, (product_id,)).fetchone()
    # Query 2: Count the number of services associated with the product
    service_count_query = """
    SELECT COUNT(*) as service_count  FROM service_data WHERE customer_product_id in (SELECT customer_product_id FROM customer_product
    WHERE product_id =?)
    """
    service_count_data = conn.execute(service_count_query, (product_id,)).fetchone()
    # Query 3: List of customers using the product
    customers_query = """
      SELECT customer_name FROM customers
    WHERE customer_id in (SELECT customer_id FROM customer_product
    WHERE product_id =?)
    """
    customers_data = conn.execute(customers_query, (product_id,)).fetchall()
    print("Product Data:", product_data)
    print("Service Count Data:", service_count_data)
    print("Customer Data:", customers_data)
    conn.close()
    if product_data is None:
        abort(404)
    return render_template('products/product_details.html', product=product_data, service_count=service_count_data, customers=customers_data)
####End of Products##

##Customers
@app.route('/customers')
def customers():
    search_query = request.args.get('search', '')
    show_all = request.args.get('show_all', 'false').lower() == 'true'
    conn = get_db_connection()
    
    if conn is None:
        flash('Database connection failed!', 'error')
        return render_template('error_page.html')  # Create an error page template

    cursor = conn.cursor()
    customers = []
    try:
        if show_all:
            query = "SELECT customer_id, customer_name, customer_since FROM customers"
            customers = conn.execute(query).fetchall()
        elif search_query:
            query = "SELECT customer_id, customer_name, customer_since FROM customers WHERE customer_name LIKE ?"
            customers = conn.execute(query, ('%' + search_query + '%',)).fetchall()
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
    finally:
        conn.close()

    return render_template('customers/customers.html', customers=customers, search=search_query, show_all=show_all)


@app.route('/add_customer', methods=('GET', 'POST'))
def add_customer():
    if request.method == 'POST':
        customer_name = request.form['customer_name'] 
        customer_since = request.form['customer_since']
        if not customer_name or not customer_since:
            flash('Customer Name and Customer Since are required!', 'error')
            return redirect(url_for('customers')) 
        conn = get_db_connection()
        conn.execute('INSERT INTO customers (customer_name,customer_since) VALUES (?,?)', (customer_name,customer_since))
        conn.commit()
        conn.close()
        flash('New Customer added successfully!', 'success')
        
    return render_template('customers/add_customer.html')

@app.route('/customer/<int:customer_id>')
def customer_details(customer_id):
    conn = get_db_connection()
    # Query 1: Fetch Customer details
    customer_query = "SELECT customer_id,customer_name, customer_since FROM customers WHERE customer_id = ?"
    customer_data = conn.execute(customer_query, (customer_id,)).fetchone()
    # Query 2: Count the number of services associated with the customer
    service_count_query = """
        SELECT COUNT(*) as service_count FROM service_data WHERE customer_product_id in (SELECT customer_product_id FROM customer_product
    WHERE customer_id =?)
    """
    service_count_data = conn.execute(service_count_query, (customer_id,)).fetchone()
    # Query 3: List of products using the customer
    products_query = """
      SELECT product_name FROM products
    WHERE product_id in (SELECT product_id FROM customer_product
    WHERE customer_id =?)
    """
    products_data = conn.execute(products_query, (customer_id,)).fetchall()
    print("Customer Data:", customer_data)
    print("Service Count Data:", service_count_data)
    print("Customer Data:", customer_data)
    conn.close()
    if customer_data is None:
        abort(404)
    return render_template('customers/customer_details.html', customer=customer_data, service_count=service_count_data, products=products_data)

####End of Customers##

##Employees
@app.route('/employees')
def employees():
    conn = get_db_connection()
    data = conn.execute('SELECT * FROM employees').fetchall()
    column_names = [description[0] for description in conn.execute('SELECT * FROM employees').description]
    conn.close()
    return render_template('employees/employees.html',data=data,column_names  =column_names)

@app.route('/add_employee', methods=('GET', 'POST'))
def add_employee():
    if request.method == 'POST':
        name = request.form['name'] 
        Designation = request.form['Designation'] 
        conn = get_db_connection()
        conn.execute('INSERT INTO employees (name,Designation) VALUES (?,?)', (name,Designation))
        conn.commit()
        conn.close()
        return redirect(url_for('employee'))
    return render_template('employees/add_employee.html')
####End of Employees##

##Spares
@app.route('/spares')
def spares():
    conn = get_db_connection()
    data = conn.execute('SELECT * FROM spares').fetchall()
    column_names = [description[0] for description in conn.execute('SELECT * FROM spares').description]
    conn.close()
    return render_template('spares/spares.html',data=data,column_names  =column_names)

@app.route('/add_spares', methods=('GET', 'POST'))
def add_spares():
    if request.method == 'POST':
        spare_name = request.form['spare_name'] 
        conn = get_db_connection()
        conn.execute('INSERT INTO spares (spare_name) VALUES (?)', (spare_name,))
        conn.commit()
        conn.close()
        return redirect(url_for('spares'))
    return render_template('spares/add_spares.html')
##End of Spares

##Service Data
@app.route('/service_data')
def service_data():
    conn = get_db_connection()
    data = conn.execute("""
    select sd.service_data_record_id,c.customer_name,p.product_name,p.model_number,
    sd.issue_date,sd.status,sd.is_warranty,e.name from service_data sd  inner join customer_product cp on cp.customer_product_id=sd.customer_product_id  
    inner join customers c on c.customer_id=cp.customer_id inner join
    products p on p.product_id=cp.product_id inner join
    employees e on e.employee_id=sd.assigned_to
    order by status desc
        """).fetchall()
    conn.close()
    return render_template('service_data/service_data.html',data=data)

@app.route('/add_service_data', methods=('GET', 'POST'))
def add_service_data():
    if request.method == 'POST':
        customer_product_id = request.form.get('customer_product_id') 
        if not customer_product_id:
            flash('customer_product_id is required!', 'error')
            return redirect(url_for('add_service_data'))
        try: 
            conn = get_db_connection()
            conn.execute('INSERT INTO service_data (customer_product_id) VALUES (?)', (customer_product_id,))
            conn.commit()
            conn.close()
            
            # Send email notification
            send_email_notification(customer_product_id)

            #folder_id = '1iY6aTwhOoSQULYsPds0V6fVbXg1uUOsT'  # Replace with your actual folder ID
            #export_data_to_google_drive(folder_id)
            
            # Flash success message
            flash('Service record added and email sent successfully!', 'success')
        except Exception as e:
            flash(f"An error occurred: {str(e)}", 'error')

        # Redirect to the same URL to show the flash message
        return redirect(url_for('add_service_data'))

    # Handle GET request: render the form
    return render_template('service_data/add_service_data.html')

def send_email_notification(customer_product_id):
    msg = Message('New Service Record Added',
                  recipients=['Thilipkumar.Dhamodaran@gmail.com'])  # Replace with recipient email
    msg.body = f'A new service record has been added for Customer Product ID: {customer_product_id}.'
    mail.send(msg)

@app.route('/export_to_drive')
def export():
    folder_id = '1iY6aTwhOoSQULYsPds0V6fVbXg1uUOsT'  # Replace with your actual folder ID
    export_data_to_google_drive(folder_id)
    return redirect(url_for('home'))

def export_data_to_google_drive(folder_id, file_name='service_data.xlsx'):
    conn = get_db_connection()
    df = pd.read_sql_query('SELECT * FROM service_data', conn)
    conn.close()

    excel_file = io.BytesIO()
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='ServiceData')
    excel_file.seek(0)

    service = build('drive', 'v3', credentials=credentials)

    query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
    response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = response.get('files', [])

    media = MediaIoBaseUpload(
        excel_file,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        resumable=True
    )

    try:
        if files:
            file_id = files[0]['id']
            updated_file = service.files().update(
                fileId=file_id,
                media_body=media
            ).execute()
            for file in files[1:]:
                service.files().delete(fileId=file['id']).execute()
        else:
            file_metadata = {
                'name': file_name,
                'mimeType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'parents': [folder_id]
            }
            new_file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()

        flash("File uploaded successfully to Google Drive.", 'success')
    except Exception as e:
        flash(f"An error occurred during file upload: {str(e)}", 'error')

@app.route('/service_records/all')
def all_service_records():
    conn = get_db_connection()
    data = conn.execute("""
        SELECT sd.service_data_record_id, c.customer_name, p.product_name, p.model_number,
        sd.issue_date, sd.status, sd.is_warranty, e.name
        FROM service_data sd
        INNER JOIN customer_product cp ON cp.customer_product_id = sd.customer_product_id
        INNER JOIN customers c ON c.customer_id = cp.customer_id
        INNER JOIN products p ON p.product_id = cp.product_id
        INNER JOIN employees e ON e.employee_id = sd.assigned_to
        ORDER BY sd.issue_date DESC
    """).fetchall()
    conn.close()
    #return render_template('service_data/all_service_records.html', data=data)
    return jsonify([dict(row) for row in data])

@app.route('/service_records/by_product/<product_name>')
def service_records_by_product(product_name):
    conn = get_db_connection()
    data = conn.execute("""
        SELECT sd.service_data_record_id, c.customer_name, p.product_name, p.model_number,
        sd.issue_date, sd.status, sd.is_warranty, e.name
        FROM service_data sd
        INNER JOIN customer_product cp ON cp.customer_product_id = sd.customer_product_id
        INNER JOIN customers c ON c.customer_id = cp.customer_id
        INNER JOIN products p ON p.product_id = cp.product_id
        INNER JOIN employees e ON e.employee_id = sd.assigned_to
        WHERE p.product_name LIKE ?
        ORDER BY sd.issue_date DESC
    """, (f"%{product_name}%",)).fetchall()
    conn.close()
    #return render_template('service_data/service_records_by_product.html', data=data)
    return jsonify([dict(row) for row in data])

@app.route('/service_records/by_company/<customer_name>')
def service_records_by_company(customer_name):
    conn = get_db_connection()
    data = conn.execute("""
        SELECT sd.service_data_record_id, c.customer_name, p.product_name, p.model_number,
        sd.issue_date, sd.status, sd.is_warranty, e.name
        FROM service_data sd
        INNER JOIN customer_product cp ON cp.customer_product_id = sd.customer_product_id
        INNER JOIN customers c ON c.customer_id = cp.customer_id
        INNER JOIN products p ON p.product_id = cp.product_id
        INNER JOIN employees e ON e.employee_id = sd.assigned_to
        WHERE c.customer_name LIKE ?
        ORDER BY sd.issue_date DESC
    """, (f"%{customer_name}%",)).fetchall()
    conn.close()
    #return render_template('service_data/service_records_by_company.html', data=data)
    return jsonify([dict(row) for row in data])

@app.route('/service_records/open_status')
def service_records_open_status():
    conn = get_db_connection()
    data = conn.execute("""
        SELECT sd.service_data_record_id, c.customer_name, p.product_name, p.model_number,
        sd.issue_date, sd.status, sd.is_warranty, e.name
        FROM service_data sd
        INNER JOIN customer_product cp ON cp.customer_product_id = sd.customer_product_id
        INNER JOIN customers c ON c.customer_id = cp.customer_id
        INNER JOIN products p ON p.product_id = cp.product_id
        INNER JOIN employees e ON e.employee_id = sd.assigned_to
        WHERE sd.status <>'Closed'
        ORDER BY sd.issue_date DESC
    """).fetchall()
    conn.close()
    #return render_template('service_data/service_records_open_status.html', data=data)
    return jsonify([dict(row) for row in data])

##End of Service Data

#Export Tab 
@app.route('/export/<table_name>')
def export_data(table_name):
    conn = get_db_connection()
    # Parameterize the query to use the table name
    query = f'SELECT * FROM {table_name}'
    data = conn.execute(query).fetchall()
    # Fetch column names
    columns = [column[0] for column in conn.execute(query).description]
    conn.close()
    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=columns)
    # Use a BytesIO buffer to create the Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    # Rewind the buffer
    output.seek(0)
    # Send the file to the user
    return send_file(output, as_attachment=True, download_name=f'{table_name}_data_export.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

if __name__ == '__main__':
    app.run(debug=True)