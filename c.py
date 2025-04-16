import pandas as pd

# Load the CSV file
file_path = '/home/luis/Documents/Amber/Folder/top1_documents.csv'
df = pd.read_csv(file_path)

# Reorder the columns (adjust the order as needed)
new_column_order = ['Top1_Document', 'Question']  # Replace with actual column names
df = df[new_column_order]

# Save the reordered DataFrame back to a CSV file
output_path = '/home/luis/Documents/Amber/Folder/top1_documents_reordered.csv'
df.to_csv(output_path, index=False)

print(f"Reordered CSV saved to {output_path}")