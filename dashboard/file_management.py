"""
File Management System for Trading Dashboard
Add this functionality to your dashboard
"""

import streamlit as st
import pandas as pd
import json
from io import BytesIO
import xlsxwriter
from datetime import datetime
import zipfile
import os
from trade_logger import TradeLogger

def create_import_export_page():
    """
    Create dedicated import/export page
    """
    st.title("📁 Data Import/Export Manager")
    
    tab1, tab2, tab3 = st.tabs(["📥 Import Data", "📤 Export Data", "🔄 Backup/Restore"])
    
    with tab1:
        import_data_section()
    
    with tab2:
        export_data_section()
    
    with tab3:
        backup_restore_section()

def import_data_section():
    """
    Import various data formats
    """
    st.subheader("📥 Import Trading Data")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file to import",
        type=['csv', 'xlsx', 'json'],
        help="Upload backtest results, historical trades, or scanner configurations"
    )
    
    if uploaded_file is not None:
        # Determine file type
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        try:
            # Read file based on type
            if file_type == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_type == 'xlsx':
                df = pd.read_excel(uploaded_file)
            elif file_type == 'json':
                data = json.load(uploaded_file)
                df = pd.DataFrame(data)
            
            # Show preview
            st.success(f"Successfully loaded {len(df)} records")
            st.dataframe(df.head(10))
            
            # Import options
            col1, col2 = st.columns(2)
            
            with col1:
                import_type = st.selectbox(
                    "Import as:",
                    ["Backtest Results", "Historical Trades", "Scanner Config", "Trade Opportunities"]
                )
            
            with col2:
                merge_option = st.radio(
                    "If duplicates exist:",
                    ["Skip", "Replace", "Append"]
                )
            
            # Validate data
            if st.button("Validate Data"):
                validation_results = validate_import_data(df, import_type)
                
                if validation_results['valid']:
                    st.success(f"✅ Data validation passed! Ready to import {validation_results['record_count']} records")
                    
                    # Show field mapping
                    st.write("**Field Mapping:**")
                    mapping_df = pd.DataFrame(
                        validation_results['field_mapping'].items(),
                        columns=['File Column', 'Database Column']
                    )
                    st.dataframe(mapping_df)
                else:
                    st.error(f"❌ Validation failed: {validation_results['error']}")
                    st.write("**Missing required fields:**", validation_results['missing_fields'])
            
            # Import button
            if st.button("🚀 Import to Database", type="primary"):
                with st.spinner("Importing data..."):
                    result = import_to_database(df, import_type, merge_option)
                    
                    if result['success']:
                        st.success(f"✅ Successfully imported {result['imported']} records!")
                        st.balloons()
                    else:
                        st.error(f"❌ Import failed: {result['error']}")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def export_data_section():
    """
    Export data in various formats
    """
    st.subheader("📤 Export Trading Data")
    
    # Export options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        export_type = st.selectbox(
            "Data to export:",
            ["Trade Opportunities", "Scan Results", "Performance Report", 
             "Backtest Results", "Complete Database Dump"]
        )
    
    with col2:
        date_range = st.selectbox(
            "Date range:",
            ["Today", "Last 7 Days", "Last 30 Days", "All Time", "Custom"]
        )
        
        if date_range == "Custom":
            start_date = st.date_input("Start date")
            end_date = st.date_input("End date")
    
    with col3:
        file_format = st.selectbox(
            "Export format:",
            ["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)", "PDF Report"]
        )
    
    # Additional filters
    with st.expander("Advanced Filters"):
        col1, col2 = st.columns(2)
        
        with col1:
            min_probability = st.slider("Min Probability %", 0, 100, 0)
            scanners = st.multiselect(
                "Scanners:",
                ["BB Scanner", "ICT", "Wyckoff", "Elliott", "All"]
            )
        
        with col2:
            timeframes = st.multiselect(
                "Timeframes:",
                ["1M", "5M", "15M", "1H", "4H", "Daily", "Weekly", "All"]
            )
            symbols = st.text_input("Symbols (comma-separated):", placeholder="BTC/USDT,ETH/USDT")
    
    # Preview data
    if st.button("Preview Data"):
        preview_data = get_export_data(export_type, date_range, filters={
            'min_probability': min_probability,
            'scanners': scanners,
            'timeframes': timeframes,
            'symbols': symbols.split(',') if symbols else []
        })
        
        if not preview_data.empty:
            st.write(f"**Preview ({len(preview_data)} records):**")
            st.dataframe(preview_data.head(20))
        else:
            st.info("No data found matching your criteria")
    
    # Export button
    if st.button("📥 Generate Export", type="primary"):
        with st.spinner("Generating export..."):
            export_data = get_export_data(export_type, date_range, filters={
                'min_probability': min_probability,
                'scanners': scanners,
                'timeframes': timeframes,
                'symbols': symbols.split(',') if symbols else []
            })
            
            if not export_data.empty:
                # Generate file
                if file_format == "Excel (.xlsx)":
                    file_data = export_to_excel(export_data, export_type)
                    mime_type = "application/vnd.ms-excel"
                    file_extension = "xlsx"
                elif file_format == "CSV (.csv)":
                    file_data = export_data.to_csv(index=False).encode('utf-8')
                    mime_type = "text/csv"
                    file_extension = "csv"
                elif file_format == "JSON (.json)":
                    file_data = export_data.to_json(orient='records').encode('utf-8')
                    mime_type = "application/json"
                    file_extension = "json"
                
                # Download button
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{export_type.lower().replace(' ', '_')}_{timestamp}.{file_extension}"
                
                st.download_button(
                    label=f"Download {filename}",
                    data=file_data,
                    file_name=filename,
                    mime=mime_type
                )
                
                st.success(f"✅ Export ready! {len(export_data)} records")
            else:
                st.warning("No data to export")

def backup_restore_section():
    """
    Backup and restore functionality
    """
    st.subheader("🔄 Backup & Restore")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 💾 Create Backup")
        
        backup_options = st.multiselect(
            "Include in backup:",
            ["Trade Opportunities", "Scan Results", "Scanner Configs", 
             "Performance Data", "User Settings"],
            default=["Trade Opportunities", "Scan Results"]
        )
        
        if st.button("Create Full Backup", type="primary"):
            with st.spinner("Creating backup..."):
                backup_data = create_full_backup(backup_options)
                
                # Create zip file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_buffer = BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for table_name, data in backup_data.items():
                        csv_data = data.to_csv(index=False)
                        zipf.writestr(f"{table_name}.csv", csv_data)
                
                zip_buffer.seek(0)
                
                st.download_button(
                    label=f"Download Backup (backup_{timestamp}.zip)",
                    data=zip_buffer.getvalue(),
                    file_name=f"trading_backup_{timestamp}.zip",
                    mime="application/zip"
                )
                
                st.success("✅ Backup created successfully!")
    
    with col2:
        st.write("### 📥 Restore from Backup")
        
        backup_file = st.file_uploader(
            "Choose backup file",
            type=['zip'],
            help="Upload a previously created backup file"
        )
        
        if backup_file:
            # Read zip file
            with zipfile.ZipFile(backup_file, 'r') as zipf:
                file_list = zipf.namelist()
                
                st.write("**Backup contents:**")
                for file in file_list:
                    st.write(f"- {file}")
            
            restore_option = st.radio(
                "Restore option:",
                ["Merge with existing data", "Replace all data (⚠️ Warning: This will delete current data)"]
            )
            
            if st.button("Restore Backup", type="primary"):
                if restore_option == "Replace all data (⚠️ Warning: This will delete current data)":
                    st.warning("⚠️ This will DELETE all current data and replace with backup!")
                    if st.checkbox("I understand and want to proceed"):
                        with st.spinner("Restoring backup..."):
                            result = restore_from_backup(backup_file, replace=True)
                            if result['success']:
                                st.success(f"✅ Restored {result['tables']} tables successfully!")
                            else:
                                st.error(f"❌ Restore failed: {result['error']}")
                else:
                    with st.spinner("Merging backup data..."):
                        result = restore_from_backup(backup_file, replace=False)
                        if result['success']:
                            st.success(f"✅ Merged {result['records']} records successfully!")

def export_to_excel(df, sheet_name="Data"):
    """
    Export DataFrame to formatted Excel file
    """
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        
        # Add formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#667eea',
            'font_color': 'white',
            'border': 1
        })
        
        number_format = workbook.add_format({'num_format': '0.00'})
        percent_format = workbook.add_format({'num_format': '0.0%'})
        date_format = workbook.add_format({'num_format': 'yyyy-mm-dd hh:mm:ss'})
        
        # Format columns
        for idx, col in enumerate(df.columns):
            series = df[col]
            max_len = max((
                series.astype(str).map(len).max(),
                len(str(series.name))
            )) + 2
            
            worksheet.set_column(idx, idx, max_len)
            
            # Apply number formats
            if 'percent' in col.lower() or 'probability' in col.lower():
                worksheet.set_column(idx, idx, max_len, percent_format)
            elif 'price' in col.lower() or 'target' in col.lower():
                worksheet.set_column(idx, idx, max_len, number_format)
            elif 'time' in col.lower() or 'date' in col.lower():
                worksheet.set_column(idx, idx, max_len, date_format)
        
        # Format header row
        worksheet.set_row(0, 30, header_format)
        
        # Add conditional formatting for profit/loss
        if 'profit_loss_percent' in df.columns:
            pl_col = df.columns.get_loc('profit_loss_percent')
            worksheet.conditional_format(1, pl_col, len(df), pl_col, {
                'type': '3_color_scale',
                'min_color': '#FF0000',
                'mid_color': '#FFFFFF',
                'max_color': '#00FF00'
            })
    
    output.seek(0)
    return output.getvalue()

def validate_import_data(df, import_type):
    """
    Validate imported data against expected schema
    """
    validation_results = {
        'valid': True,
        'error': None,
        'missing_fields': [],
        'field_mapping': {},
        'record_count': len(df)
    }
    
    # Define required fields for each import type
    required_fields = {
        'Backtest Results': ['symbol', 'entry_time', 'exit_time', 'profit_loss'],
        'Historical Trades': ['symbol', 'entry_price', 'exit_price', 'trade_result'],
        'Trade Opportunities': ['symbol', 'probability', 'entry_price', 'stop_loss'],
        'Scanner Config': ['scanner_name', 'timeframe', 'parameters']
    }
    
    # Check required fields
    if import_type in required_fields:
        for field in required_fields[import_type]:
            # Try to find matching column (case-insensitive)
            matching_cols = [col for col in df.columns if field.lower() in col.lower()]
            
            if matching_cols:
                validation_results['field_mapping'][matching_cols[0]] = field
            else:
                validation_results['missing_fields'].append(field)
                validation_results['valid'] = False
    
    if validation_results['missing_fields']:
        validation_results['error'] = f"Missing required fields: {', '.join(validation_results['missing_fields'])}"
    
    return validation_results

def import_to_database(df, import_type, merge_option):
    """
    Import data to database with merge options
    """
    from trade_logger import TradeLogger
    
    result = {
        'success': False,
        'imported': 0,
        'skipped': 0,
        'error': None
    }
    
    try:
        logger = TradeLogger()
        
        if import_type == 'Trade Opportunities':
            # Import trade opportunities
            for _, row in df.iterrows():
                trade_data = row.to_dict()
                
                # Check if exists (simplified example)
                if merge_option == 'Skip':
                    # Check for existing record
                    existing = check_existing_trade(trade_data)
                    if existing:
                        result['skipped'] += 1
                        continue
                
                # Log trade opportunity
                scan_id = logger.log_scan_start('import', version='1.0')
                logger.log_trade_opportunity(scan_id, trade_data)
                result['imported'] += 1
        
        result['success'] = True
        logger.close()
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def get_export_data(export_type, date_range, filters):
    """
    Get data for export based on criteria
    """
    from trade_logger import TradeLogger
    
    logger = TradeLogger()
    df = pd.DataFrame()
    
    if logger.connection:
        # Build query based on export type
        if export_type == "Trade Opportunities":
            query = """
                SELECT t.*, s.scan_type, s.scan_timestamp
                FROM trade_opportunities t
                JOIN scan_results s ON t.scan_id = s.id
                WHERE 1=1
            """
            
            # Add date filter
            if date_range == "Today":
                query += " AND t.timestamp >= CURRENT_DATE"
            elif date_range == "Last 7 Days":
                query += " AND t.timestamp >= CURRENT_DATE - INTERVAL '7 days'"
            elif date_range == "Last 30 Days":
                query += " AND t.timestamp >= CURRENT_DATE - INTERVAL '30 days'"
            
            # Add probability filter
            if filters.get('min_probability', 0) > 0:
                query += f" AND t.probability >= {filters['min_probability']}"
            
            # Execute query
            logger.cursor.execute(query)
            results = logger.cursor.fetchall()
            
            if results:
                df = pd.DataFrame(results)
        
        logger.close()
    
    return df

def create_full_backup(backup_options):
    """
    Create full database backup
    """
    from trade_logger import TradeLogger
    
    backup_data = {}
    logger = TradeLogger()
    
    if logger.connection:
        # Backup each selected table
        if "Trade Opportunities" in backup_options:
            logger.cursor.execute("SELECT * FROM trade_opportunities")
            backup_data['trade_opportunities'] = pd.DataFrame(logger.cursor.fetchall())
        
        if "Scan Results" in backup_options:
            logger.cursor.execute("SELECT * FROM scan_results")
            backup_data['scan_results'] = pd.DataFrame(logger.cursor.fetchall())
        
        logger.close()
    
    return backup_data

def restore_from_backup(backup_file, replace=False):
    """
    Restore data from backup file
    """
    result = {
        'success': False,
        'tables': 0,
        'records': 0,
        'error': None
    }
    
    try:
        # Process backup file
        with zipfile.ZipFile(backup_file, 'r') as zipf:
            for filename in zipf.namelist():
                if filename.endswith('.csv'):
                    # Read CSV data
                    csv_data = zipf.read(filename)
                    df = pd.read_csv(BytesIO(csv_data))
                    
                    # Restore to database
                    table_name = filename.replace('.csv', '')
                    if replace:
                        # Clear existing data first
                        clear_table(table_name)
                    
                    # Import data
                    import_result = import_table_data(table_name, df)
                    result['records'] += import_result['imported']
                    result['tables'] += 1
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result 

# Helper functions
def check_existing_trade(trade_data):
    """Check if trade already exists in database"""
    # Implementation depends on your database schema
    return False

def clear_table(table_name):
    """Clear all data from a table"""
    # Implementation with proper safety checks
    pass

def import_table_data(table_name, df):
    """Import DataFrame to specific table"""
    # Implementation based on table schema
    return {'imported': len(df)} 