import os

from google.cloud import storage


def download_blob(bucket_name, source_blob_name, destination_file_name, project=None):
    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")


def download_folder(bucket_name, folder_name, local_folder, project=None):
    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder_name)

    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        # Create local path
        local_path = os.path.join(local_folder, os.path.relpath(blob.name, folder_name))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"Downloading {blob.name}...")
        blob.download_to_filename(local_path)


if __name__ == "__main__":
    # Set your Google Cloud project ID here or via GOOGLE_CLOUD_PROJECT env var
    project_id = os.environ.get(
        "GOOGLE_CLOUD_PROJECT", "academic-torch-476716-h3"
    )  # Replace with your actual project ID
    download_folder("tikkamasalai-models", "", "./models", project=project_id)
