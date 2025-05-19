import pandas as pd
import csv  # Add csv import for QUOTE_MINIMAL

def drop_columns(df, columns_to_drop):
    """
    Drop specified columns from the dataframe
    Args:
        df: pandas DataFrame
        columns_to_drop: list of column names to drop
    Returns:
        DataFrame with specified columns dropped
    """
    # Get existing columns that are actually in the dataframe
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    if existing_columns:
        print(f"\nDropping columns: {existing_columns}")
        df = df.drop(columns=existing_columns)
        print("Columns dropped successfully")
    else:
        print("\nNo specified columns found in the dataset")
    return df



def clean_imdb_id(id_str):
    # Remove 'tt' prefix and leading zeros, but keep it as string
    if isinstance(id_str, str) and id_str.startswith('tt'):
        return str(int(id_str[2:]))  # Remove 'tt' and convert to int then back to str to remove leading zeros
    return id_str




def merge_multiple_datasets(dataset_paths, columns_to_drop=None):
    # Initialize with the first dataset
    merged_df = pd.read_csv(dataset_paths[0], low_memory=False, 
                        encoding='utf-8',
                        on_bad_lines='skip')  # Skip problematic rows
    
    # Print columns to debug
    print(f"\nColumns in first dataset: {merged_df.columns.tolist()}")
    
    if 'id' not in merged_df.columns:
        # Try to extract ID from first column if it's malformed
        if merged_df.columns[0].strip().startswith("{'id"):
            # Reset the DataFrame by reading the CSV again with different parameters
            merged_df = pd.read_csv(dataset_paths[0], low_memory=False,
                                encoding='utf-8',
                                on_bad_lines='skip',
                                delimiter=',')
    
    # Verify that 'id' column exists
    if 'id' not in merged_df.columns:
        print(f"Available columns in first dataset: {merged_df.columns.tolist()}")
        raise ValueError("No 'id' column found in the first dataset")
    
    merged_df['id'] = merged_df['id'].astype(str)
    initial_records = len(merged_df)
    
    print(f"\nInitial dataset records: {initial_records}")
    
    # Track records for each merge
    total_records = [initial_records]
    
    # Merge remaining datasets one by one
    for i, path in enumerate(dataset_paths[1:], 1):
        # Read the next dataset
        df = pd.read_csv(path, low_memory=False,
                        encoding='utf-8',
                        on_bad_lines='skip')  # Skip problematic rows
            
        # Verify that 'id' column exists
        if 'id' not in df.columns:
            print(f"Warning: No 'id' column found in dataset {i+1}. Available columns: {df.columns.tolist()}")
            continue
            
        df['id'] = df['id'].astype(str)
        current_records = len(df)
        total_records.append(current_records)
        
        # Store previous merged records count
        previous_merged_count = len(merged_df)
        
        # Merge with accumulated result
        merged_df = pd.merge(merged_df, df, on='id', how='inner')
        current_merged_count = len(merged_df)
        
        # Calculate and display matching statistics
        print(f"\nMerge {i} statistics:")
        print(f"Dataset {i+1} records: {current_records}")
        print(f"Records after merge: {current_merged_count}")
        print(f"Matching percentage with previous merge: {(current_merged_count / previous_merged_count) * 100:.2f}%")
        print(f"Matching percentage with dataset {i+1}: {(current_merged_count / current_records) * 100:.2f}%")
    
    # Display final statistics
    print(f"\nFinal Statistics:")
    print(f"Initial dataset records: {initial_records}")
    for i, records in enumerate(total_records[1:], 1):
        print(f"Dataset {i+1} records: {records}")
    print(f"Final matched records: {len(merged_df)}")
    print(f"Overall matching percentage: {(len(merged_df) / min(total_records)) * 100:.2f}%")
    
    # Clean up imdb_id column if it exists in the final merged dataset
    if 'imdb_id' in merged_df.columns:
        print("\nCleaning up imdb_id column...")
        merged_df['imdb_id'] = merged_df['imdb_id'].apply(clean_imdb_id)
        print("imdb_id cleanup complete")
    
    # Drop specified columns if any
    if columns_to_drop:
        merged_df = drop_columns(merged_df, columns_to_drop)
    
    return merged_df