# SQL Server Customer Search Setup

This document explains how to set up the SQL Server customer search functionality.

## Prerequisites

1. **SQL Server Access**: You need access to the SQL Server database `SOCDB_EDMOND_BARADEI`
2. **ODBC Driver**: Install "ODBC Driver 17 for SQL Server" on your system
3. **Network Access**: Ensure your application server can connect to the SQL Server

## Installation

1. **Install Dependencies**:
   ```bash
   pip install pyodbc>=4.0.0
   ```

2. **Configure Connection**:
   - Copy `sql_server_config.env` to `.env` or set environment variables
   - Update the connection details in the configuration

## Configuration

### Environment Variables

Set these environment variables or add them to your `.env` file:

```bash
# SQL Server connection details
SQL_SERVER_HOST=your_server_name_or_ip
SQL_SERVER_DB=SOCDB_EDMOND_BARADEI
SQL_SERVER_USER=your_username
SQL_SERVER_PASSWORD=your_password

# Use Windows Authentication (set to 'yes' if using Windows Auth, 'no' for SQL Auth)
SQL_SERVER_TRUSTED=no
```

### Example Configurations

**SQL Server Authentication:**
```bash
SQL_SERVER_HOST=localhost
SQL_SERVER_USER=sa
SQL_SERVER_PASSWORD=YourPassword123
SQL_SERVER_TRUSTED=no
```

**Windows Authentication:**
```bash
SQL_SERVER_HOST=localhost
SQL_SERVER_TRUSTED=yes
# SQL_SERVER_USER and SQL_SERVER_PASSWORD can be left empty
```

## Testing the Connection

1. **Start your application**
2. **Test the connection** by visiting: `http://localhost:5000/api/test-sql-server-connection`
3. **Expected response**: `{"success": true, "message": "Connection successful"}`

## Usage

### In the PDF Quote Details Form

1. **Upload a DXF file** and process it
2. **Click "Generate PDF"** button
3. **In the PDF Quote Details modal**:
   - Enter a company name in the search box (e.g., "mitsu")
   - Click "Search" or press Enter
   - Browse the search results
   - **Double-click** on the desired customer to auto-fill the form

### Field Mapping

When you select a customer, the following fields are automatically filled:

- **Company Name** ← `FileName` from database
- **PSC Number** ← `FileNumber` from database  
- **VAT ID** ← `VATID` from database
- **Contact Name** ← `FirstName` from database
- **Telephone** ← `TEL` from database
- **Email** ← `EMAIL` from database
- **Company Address** ← `Address`Concatenated address fields from database

### Manual Fields

These fields must be filled manually:
- **Quotation Number** (required)
- **Payment Terms** (dropdown)
- **Delivery Method** (dropdown)
- **Delivery Place** (dropdown)
- **Delivery Date** (dropdown)
- **Validity** (dropdown)

## Troubleshooting

### Common Issues

1. **"SQL Server connection not configured"**
   - Check that environment variables are set correctly
   - Ensure `pyodbc` is installed

2. **"Connection failed"**
   - Verify SQL Server is running and accessible
   - Check network connectivity
   - Verify credentials
   - Ensure ODBC Driver 17 is installed

3. **"No customers found"**
   - Verify the database contains data in the `CustomerFile` table
   - Check that the search term matches existing company names

### Database Query

The search uses this SQL query:
```sql
SELECT DISTINCT
    FileNumber, FileName, FirstName, TEL, EMAIL, 
    LTRIM(RTRIM(CASE WHEN LTRIM(RTRIM(Main_Region)) <> '' THEN LTRIM(RTRIM(Main_Region)) ELSE '' END + 
    CASE WHEN LTRIM(RTRIM(Bldg)) <> '' THEN CASE WHEN LTRIM(RTRIM(Main_Region)) <> '' THEN ' - ' ELSE '' END + LTRIM(RTRIM(Bldg)) ELSE '' END + 
    CASE WHEN LTRIM(RTRIM(Adress_Desc)) <> '' THEN CASE WHEN LTRIM(RTRIM(Main_Region)) <> '' OR LTRIM(RTRIM(Bldg)) <> '' THEN ' - ' ELSE '' END + LTRIM(RTRIM(Adress_Desc)) ELSE '' END + 
    CASE WHEN LTRIM(RTRIM(Country_Desc)) <> '' THEN CASE WHEN LTRIM(RTRIM(Main_Region)) <> '' OR LTRIM(RTRIM(Bldg)) <> '' OR LTRIM(RTRIM(Adress_Desc)) <> '' THEN ' - ' ELSE '' END + LTRIM(RTRIM(Country_Desc)) ELSE '' END)) AS Address, 
    VATID
FROM SOCDB_EDMOND_BARADEI.dbo.CustomerFile
WHERE FileName LIKE '%search_term%'
ORDER BY FileName
```

## Security Notes

- Store database credentials securely
- Use Windows Authentication when possible
- Consider using connection pooling for production
- Ensure proper network security between application and database servers
