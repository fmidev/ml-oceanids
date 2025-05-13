# /home/ubuntu/ml-oceanids/bin/greece-obs-operations.py
#!/usr/bin/env python3
import os
import sys
import argparse
import csv
import re
import datetime

def extract_station_name(file_content, input_file=None):
    """Extract station name from file content."""
    # Dictionary of Greek to English station names
    station_mapping = {
        "ΧΙΟΣ": "Chios",
        "ΚΕΡΚΥΡΑ": "Kerkyra",
        "ΗΡΑΚΛΕΙΟ": "Heraklion",
        "Ηράκλειο": "Heraklion",  # Add lowercase version
        "ΣΤΑΘΜΟΣ: Ηράκλειο": "Heraklion",  # Add the specific format from this file
        "ΚΑΣΤΕΛΛΙ": "Kastelli",
        "ΛΗΜΝΟΣ": "Limnos",
        "Λήμνος": "Limnos",  # Add lowercase version
        "ΜΗΛΟΣ": "Milos",
        "ΜΥΚΟΝΟΣ": "Mykonos",
        "ΜΥΤΙΛΗΝΗ": "Mytilene",
        "ΝΑΞΟΣ": "Naxos",
        "ΠΑΛΑΙΟΧΩΡΑ": "Palaiochora",
        "ΑΓΙΟΣ ΕΥΣΤΡΑΤΙΟΣ": "Saint_Eustratios",
        "ΣΗΤΕΙΑ": "Siteia"
    }
    
    for line in file_content:
        for greek_name, english_name in station_mapping.items():
            if greek_name in line:
                return english_name
    
    # If we couldn't find a match, extract from filename if possible
    if input_file:
        filename = os.path.basename(input_file)
        if "_" in filename:
            station_part = filename.split("_")[0]
            return station_part.capitalize()
    
    return "Unknown"  # Default if not found

def extract_measurement_type(line):
    """Extract the type of measurement from a line."""
    if "Μέτρηση:" in line:
        # Wind speed
        if "Ανέμου" in line:
            return "wind_speed"
        
        # Temperature variations
        if "Μέγιστη" in line and "Θερμοκρασία" in line:
            return "max_temperature"
        elif "Ελάχιστη" in line and "Θερμοκρασία" in line:
            return "min_temperature"
        elif "Θερμοκρασία" in line:
            return "temperature"
        
        # Other measurements
        elif "Υγρασία" in line:
            return "humidity"
        elif "Πίεση" in line:
            return "pressure"
        elif "Βροχόπτωση" in line or "Υετός" in line:
            return "precipitation"
        
        # Extract actual measurement description
        match = re.search(r'Μέτρηση:\s*(.+?)\s*\(', line)
        if match:
            measurement = match.group(1).strip().lower().replace(' ', '_')
            return measurement
    
    return "unknown"  # Default if not found

def extract_units(line):
    """Extract measurement units from a line."""
    if "Μέτρηση:" in line:
        unit_match = re.search(r'\((.*?)\)', line)
        if unit_match:
            unit = unit_match.group(1)
            if unit == "knots":
                return "knots"
            elif unit == "oC":
                return "celsius"
            elif unit == "hPa":
                return "hpa"
            elif unit == "%" or unit == "%RH":
                return "percent"
            elif unit == "mm":
                return "mm"
            return unit
    return "unknown"

def convert_file(input_file, output_file):
    """Process Greek meteorological station data file."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            file_content = f.readlines()
        
        # Initial station extraction (only gets the first one)
        station = extract_station_name(file_content, input_file)
        
        with open(output_file, 'w', newline='') as out_f:
            writer = csv.writer(out_f)
            # Write header
            writer.writerow(["timestamp", "station", "measurement", "type", "unit"])
            
            # Initialize variables for tracking measurement sections
            current_measurement_type = "unknown"
            current_unit = "unknown"
            data_section = False
            processed_rows = 0
            
            # Process each line
            for line in file_content:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Check if this is a measurement type line
                if "Μέτρηση:" in line:
                    current_measurement_type = extract_measurement_type(line)
                    current_unit = extract_units(line)
                    data_section = False
                    continue
                
                # Check if this is a column header line
                if 'ΕΤΟΣ' in line or 'Όνομα Σταθμού' in line or 'Προσοχή' in line:
                    data_section = True
                    continue
                
                # Process data lines
                parts = line.split(',')
                if len(parts) >= 4 and re.match(r'"?\d{4}"?', parts[0]):
                    year = parts[0].strip('"')
                    month = parts[1].strip('"')
                    day = parts[2].strip('"')
                    value = parts[3].strip('"')
                    
                    # Handle missing or invalid values
                    if value == "*" or not value:
                        continue
                    
                    # Convert comma decimal separator to dot
                    value = value.replace(',', '.')
                    try:
                        value = float(value)
                    except ValueError:
                        print(f"Warning: Could not convert value '{value}' to float")
                        continue
                    
                    # Create ISO timestamp format (YYYY-MM-DD)
                    try:
                        timestamp = f"{year}-{int(month):02d}-{int(day):02d}"
                        # Validate date
                        datetime.datetime.strptime(timestamp, '%Y-%m-%d')
                        
                        # Write row in target format [timestamp, station, measurement, type, unit]
                        writer.writerow([timestamp, station, str(value), current_measurement_type, current_unit])
                        processed_rows += 1
                    except ValueError as e:
                        print(f"Warning: Invalid date {year}-{month}-{day}: {e}")
            
            print(f"Processed {processed_rows} data rows from file: {os.path.basename(input_file)}")
            print(f"Output written to: {output_file}")
            return True
            
    except Exception as e:
        print(f"Error processing file {input_file}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert Greek meteorological station file to standard format.')
    parser.add_argument('-i', '--input_file', required=True,
                        help='Input file path for Greek station data (CSV format)')
    parser.add_argument('-o', '--output_file', required=True,
                        help='Output file path for the converted data')
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
    
    # Process the single file
    input_file = args.input_file  # Define for extract_station_name function
    success = convert_file(args.input_file, args.output_file)
    if success:
        print(f"Conversion complete. Output written to {args.output_file}")
    else:
        print("Conversion failed.")
        sys.exit(1)

if __name__ == '__main__':
    main()
