from _config import DATABASE, DRIVER, SERVER
import pyodbc

def create_connection_string(database_type):
    """This function is used to create either an on prem database connection string on an
       azure cloud database connection string"""
    
    if database_type == 'sql_server':
        CERTIFICATE = ';Encrypt=yes;TrustServerCertificate=yes;Trusted_Connection=yes;'
        CONNECTION_STRING = f'Driver={DRIVER};Server={SERVER};Database={DATABASE}' + CERTIFICATE
        
    if database_type == 'azure':
        CERTIFICATE = 'Encrypt=yes;TrustServerCertificate=no;Connection Timeout=999;'
        CONNECTION_STRING = f'Driver={DRIVER};Server={SERVER},1433;Database={DATABASE};Uid={USERNAME};Pwd={PASSWORD};' + CERTIFICATE
    
    return CONNECTION_STRING

def init_database_connection(database_type):
    """This function is used to create a database connection and cursor that
       can be used to execute queries"""
    
    CONNECTION_STRING = create_connection_string(database_type)
    
    connection = pyodbc.connect(CONNECTION_STRING)
    cursor = connection.cursor()

    return connection, cursor