import json

from app import app


def export_openapi_schema():
    """Generates and saves the openapi.json schema file."""
    
    print("Exporting OpenAPI schema...")
    
    # Get the schema as a Python dict
    openapi_schema = app.openapi()
    
    # Save it to a file
    destination_path = "src/backend/openapi.json"
    with open(destination_path, "w") as f:
        json.dump(openapi_schema, f, indent=2)
        
    print(f"Successfully exported to {destination_path}")

if __name__ == "__main__":
    export_openapi_schema()