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
        
        number_format = workbook.add_format({
            'num_format': '#,##0.00',
            'border': 1
        })
        
        percent_format = workbook.add_format({
            'num_format': '0.00%',
            'border': 1
        })
        
        # Apply formats
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Auto-adjust column widths
        for i, col in enumerate(df.columns):
            max_len = max(
                df[col].astype(str).map(len).max(),
                len(col)
            ) + 2
            worksheet.set_column(i, i, max_len)
    
    output.seek(0)
    return output.getvalue()

def validate_import_data(df, import_type):
    """
    Validate imported data based on type
    """
    required_fields = {
        "Trade Opportunities": ["symbol", "probability", "entry_price", "stop_loss", "target_1"],
        "Backtest Results": ["symbol", "entry_price", "exit_price", "profit_loss", "timestamp"],
        "Scan Results": ["scan_type", "symbol", "timestamp"],
        "Historical Trades": ["symbol", "entry_price", "exit_price", "profit_loss_percent"]
    }
    
    required = required_fields.get(import_type, [])
    missing = [field for field in required if field not in df.columns]
    
    field_mapping = {}
    for col in df.columns:
        field_mapping[col] = col  # Simple mapping for now
    
    return {
        'valid': len(missing) == 0,
        'record_count': len(df),
        'missing_fields': missing,
        'field_mapping': field_mapping,
        'error': f"Missing required fields: {missing}" if missing else None
    }

def import_to_database(df, import_type, merge_option):
    """
    Import data to database
    """
    logger = TradeLogger()
    
    if not logger.connection:
        return {'success': False, 'error': 'Database connection failed'}
    
    try:
        imported = 0
        
        if import_type == "Trade Opportunities":
            for _, row in df.iterrows():
                logger.cursor.execute("""
                    INSERT INTO trade_opportunities 
                    (symbol, probability, entry_price, stop_loss, target_1, 
                     risk_reward_ratio, timestamp, scan_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, timestamp) DO NOTHING
                """, (
                    row.get('symbol', ''),
                    row.get('probability', 0),
                    row.get('entry_price', 0),
                    row.get('stop_loss', 0),
                    row.get('target_1', 0),
                    row.get('risk_reward_ratio', 0),
                    row.get('timestamp', datetime.now()),
                    1  # Default scan_id
                ))
                imported += 1
        
        logger.connection.commit()
        return {'success': True, 'imported': imported}
    
    except Exception as e:
        logger.connection.rollback()
        return {'success': False, 'error': str(e)}

def get_export_data(export_type, date_range, filters=None):
    """
    Get data for export based on type and filters
    """
    logger = TradeLogger()
    
    if not logger.connection:
        return pd.DataFrame()
    
    # Build date filter
    date_filters = {
        "Today": "1 day",
        "Last 7 Days": "7 days", 
        "Last 30 Days": "30 days",
        "All Time": "10 years"
    }
    
    date_filter = date_filters.get(date_range, "30 days")
    
    try:
        if export_type == "Trade Opportunities":
            query = """
                SELECT 
                    t.symbol,
                    t.probability,
                    t.entry_price,
                    t.stop_loss,
                    t.target_1,
                    t.risk_reward_ratio,
                    t.timestamp,
                    s.scan_type
                FROM trade_opportunities t
                JOIN scan_results s ON t.scan_id = s.id
                WHERE t.timestamp > NOW() - INTERVAL %s
            """
            
            if filters and filters.get('min_probability', 0) > 0:
                query += f" AND t.probability >= {filters['min_probability']}"
            
            query += " ORDER BY t.timestamp DESC"
            
            logger.cursor.execute(query, (date_filter,))
            results = logger.cursor.fetchall()
            
            return pd.DataFrame(results)
        
        elif export_type == "Scan Results":
            query = """
                SELECT 
                    scan_type,
                    symbols_scanned,
                    opportunities_found,
                    scan_timestamp,
                    execution_time
                FROM scan_results
                WHERE scan_timestamp > NOW() - INTERVAL %s
                ORDER BY scan_timestamp DESC
            """
            
            logger.cursor.execute(query, (date_filter,))
            results = logger.cursor.fetchall()
            
            return pd.DataFrame(results)
        
        else:
            return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Error getting export data: {str(e)}")
        return pd.DataFrame()

def create_full_backup(backup_options):
    """
    Create full backup of selected data
    """
    logger = TradeLogger()
    backup_data = {}
    
    if not logger.connection:
        return backup_data
    
    try:
        if "Trade Opportunities" in backup_options:
            logger.cursor.execute("SELECT * FROM trade_opportunities")
            backup_data['trade_opportunities'] = pd.DataFrame(logger.cursor.fetchall())
        
        if "Scan Results" in backup_options:
            logger.cursor.execute("SELECT * FROM scan_results")
            backup_data['scan_results'] = pd.DataFrame(logger.cursor.fetchall())
        
        return backup_data
    
    except Exception as e:
        st.error(f"Backup error: {str(e)}")
        return backup_data

def restore_from_backup(backup_file, replace=False):
    """
    Restore data from backup file
    """
    logger = TradeLogger()
    
    if not logger.connection:
        return {'success': False, 'error': 'Database connection failed'}
    
    try:
        restored_tables = 0
        total_records = 0
        
        with zipfile.ZipFile(backup_file, 'r') as zipf:
            for filename in zipf.namelist():
                if filename.endswith('.csv'):
                    table_name = filename.replace('.csv', '')
                    
                    # Read CSV from zip
                    with zipf.open(filename) as f:
                        df = pd.read_csv(f)
                    
                    if replace:
                        # Clear existing data
                        logger.cursor.execute(f"DELETE FROM {table_name}")
                    
                    # Insert new data
                    for _, row in df.iterrows():
                        # This is a simplified insert - you'd need proper column mapping
                        columns = ', '.join(df.columns)
                        placeholders = ', '.join(['%s'] * len(df.columns))
                        
                        logger.cursor.execute(f"""
                            INSERT INTO {table_name} ({columns})
                            VALUES ({placeholders})
                        """, tuple(row.values))
                    
                    restored_tables += 1
                    total_records += len(df)
        
        logger.connection.commit()
        return {'success': True, 'tables': restored_tables, 'records': total_records}
    
    except Exception as e:
        logger.connection.rollback()
        return {'success': False, 'error': str(e)} 