# Knowledge Database Interaction with Weaviate

This prototype interacts with a Knowledge Database built in **Weaviate**, a vector database designed for storing and retrieving interconnected structured objects and their relationships.

## Database Structure

The database organizes information into a hierarchical structure as follows:

- **QueryEntityInfo** (query) → linked to → **Documents** (via `inDocs`)
- **Documents** → linked to → **Pages** (via `hasPage`)
- **Pages** → linked to → **Results** (via `hasResult`)

## How It Works

The provided code recursively navigates these relationships by:

1. **Querying the main entity**: Starts by querying the `QueryEntityInfo` node.
2. **Traversing to associated entities**: The traversal then moves to related `Documents`, `Pages`, and `Results`.
3. **Extracting information**: Each entity’s information is extracted and displayed through its **UUIDs** using `weaviate.util.get_valid_uuid`.

## Purpose

This process allows for a complete retrieval of related knowledge starting from a **QueryEntityInfo** node, forming a dynamic and flexible **knowledge graph** traversal.

## Example

```python
import weaviate
from weaviate.util import get_valid_uuid

# Initialize Weaviate client
client = weaviate.Client("http://localhost:8080")

# Example recursive function to traverse the graph
def retrieve_entity_info(query_entity_uuid):
    # Query the main entity
    query_entity = client.data_object.get(uuid=query_entity_uuid)
    
    # Traverse related documents, pages, and results
    documents = query_entity['inDocs']
    for doc in documents:
        doc_info = client.data_object.get(uuid=doc['uuid'])
        pages = doc_info['hasPage']
        
        for page in pages:
            page_info = client.data_object.get(uuid=page['uuid'])
            results = page_info['hasResult']
            
            for result in results:
                result_info = client.data_object.get(uuid=result['uuid'])
                print(result_info)

# Start traversal from a given QueryEntityInfo UUID
retrieve_entity_info("query_entity_uuid_here")
