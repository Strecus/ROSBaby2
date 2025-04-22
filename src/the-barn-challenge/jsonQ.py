import json

# Function to read and parse a JSON file
def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)  # Load JSON data from the file
            return data
    except FileNotFoundError:
        # Handle the case where the file does not exist
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        # Handle the case where the file is not a valid JSON
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
    except Exception as e:
        # Handle any other unexpected errors
        print(f"An unexpected error occurred: {e}")

# Function to remove the first element from a JSON array in a file
def dequeue_json_data(file_path, cmd):
    data = read_json_file(file_path)  # Read the JSON data from the file
    if data is not None:
        if len(data) == 1:
            # If the array has only one element, clear it
            data = []
        else:
            # Remove the first element from the array
            data = data[1::]
        try:
            with open(file_path, 'w') as f:
                # Write the updated JSON data back to the file
                json.dump(data, f, indent=2)  # indent=2 for pretty formatting
        except FileNotFoundError:
            # Handle the case where the file does not exist
            print(f"Error: The file '{file_path}' was not found.")
        except json.JSONDecodeError:
            # Handle the case where the file is not a valid JSON
            print(f"Error: The file '{file_path}' is not a valid JSON file.")
        except Exception as e:
            # Handle any other unexpected errors
            print(f"An unexpected error occurred: {e}")

# Function to add a new element to a JSON array in a file
def enqueue_json_data(file_path, cmd):
    data = read_json_file(file_path)  # Read the JSON data from the file
    if data is not None:
        # Append the new command to the JSON array
        data.append(cmd)
        try:
            with open(file_path, 'w') as f:
                # Write the updated JSON data back to the file
                json.dump(data, f, indent=2)  # indent=2 for pretty formatting
        except FileNotFoundError:
            # Handle the case where the file does not exist
            print(f"Error: The file '{file_path}' was not found.")
        except json.JSONDecodeError:
            # Handle the case where the file is not a valid JSON
            print(f"Error: The file '{file_path}' is not a valid JSON file.")
        except Exception as e:
            # Handle any other unexpected errors
            print(f"An unexpected error occurred: {e}")
