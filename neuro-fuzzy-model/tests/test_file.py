file_path = "C:/Users/East-Sound/Desktop/combined_data_1.txt"

with open(file_path, 'r') as file:
    for i in range(10):
        line = file.readline()
        if not line:  # If less than 10 lines exist
            break
        print(line.strip())
