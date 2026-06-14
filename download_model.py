import os
import requests

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    print("Downloading chunks...")
    downloaded = 0
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                downloaded += len(chunk)
                # Print progress every ~5MB
                if downloaded % (5 * 1024 * 1024) < CHUNK_SIZE:
                    print(f"Downloaded: {downloaded / (1024 * 1024):.1f} MB")
    print("Done writing file.")

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    # Headers to mimic a real browser
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    print("Connecting to Google Drive...")
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        print(f"Warning page bypass token found: {token}")
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    else:
        print("No warning page token required or could be retrieved directly.")
        
    save_response_content(response, destination)

def download_model():
    file_id = '1LJG_ITCWWtriLC5NPrWxIDwekWbhU_Rj'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'models')
    output_path = os.path.join(output_dir, 'bt_resnet50_model.pt')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    if os.path.exists(output_path) and os.path.getsize(output_path) > 10 * 1024 * 1024:
        print(f"Model file already exists at {output_path} and is valid size. Skipping download.")
        return
        
    print(f"Downloading model weights from Google Drive to {output_path}...")
    try:
        download_file_from_google_drive(file_id, output_path)
        print("Download completed successfully!")
    except Exception as e:
        print(f"Error downloading the model: {e}")
        print("Please check your internet connection or download manually from: https://drive.google.com/file/d/1LJG_ITCWWtriLC5NPrWxIDwekWbhU_Rj/view")

if __name__ == '__main__':
    download_model()
