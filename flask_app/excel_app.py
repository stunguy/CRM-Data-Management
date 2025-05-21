from flask import Flask, render_template, request, redirect, url_for,send_file
import pandas as pd
import matplotlib.pyplot as plt
import os
import io

app = Flask(__name__)

EXCEL_FILE_PATH = 'C:/Users/tdhamodaran/Downloads/Sample_Retex Service Inward Entry.xlsx'


@app.route('/')
def index():
    # Read data from Excel file
    if os.path.exists(EXCEL_FILE_PATH):
        df = pd.read_excel(EXCEL_FILE_PATH,sheet_name='Sheet1' ,engine='openpyxl')
        data = df.to_dict(orient='records')

    # Read dropdown options from another sheet
        df_dropdown = pd.read_excel(EXCEL_FILE_PATH, sheet_name='Sheet2', engine='openpyxl')
        dropdown_options = df_dropdown['Products'].tolist()
    else:
        data = []
        dropdown_options = []
    return render_template('index_xl.html', data=data,dropdown_options=dropdown_options)

@app.route('/plot.png')
def plot_png():
    df = pd.read_excel(EXCEL_FILE_PATH,sheet_name='Sheet1' ,engine='openpyxl')
    Date_Received = df['Date Received'].astype(str)
    Service_Charge = df['Service Charge'].astype(float)
 # Create a bar chart
    fig, ax = plt.subplots()
    ax.bar(Date_Received, Service_Charge)
    ax.set_title("Bar Chart from Excel Data")
    ax.set_xlabel("Date Received")
    ax.set_ylabel("Service Charge")

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

@app.route('/add', methods=['POST'])
def add_data():
    # Get form data
    Customer_name = request.form['Customer name']
    DC_No = request.form['DC No']
    Product_Name = request.form['Product Name']

 # Create a new DataFrame with the new data
    new_data = pd.DataFrame({
        'Customer name': [Customer_name],
        'DC No': [DC_No],
        'Product Name': [Product_Name]
    })

    
    # Append new data to the existing Excel file
    if os.path.exists(EXCEL_FILE_PATH):
        df_main = pd.read_excel(EXCEL_FILE_PATH,sheet_name='Sheet1' ,engine='openpyxl')
        df_main = pd.concat([df_main, new_data], ignore_index=True)
    else:
        df_main = new_data

    # Save the updated DataFrame back to the Excel file
    with pd.ExcelWriter(EXCEL_FILE_PATH, engine='openpyxl') as writer:
     df_main.to_excel(writer, sheet_name='Sheet1', index=False)
     # Ensure the dropdown sheet is preserved
     df_dropdown = pd.read_excel(EXCEL_FILE_PATH, sheet_name='Sheet2', engine='openpyxl')
     df_dropdown.to_excel(writer, sheet_name='Sheet2', index=False)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
