import weaviate
import json

client = weaviate.Client(
    url = "https://nlri-grsrbdwa.weaviate.network/",  # Replace with your endpoint
   
)

#client.schema.delete_class('PageInfo')
#delete all
#client.schema.delete_all()

schema = [
    {
    "class": "QueryEntityInfo",
    "description": "store query property and results mapped to the query",
    "vectorizer": "none",
    "properties": [
        {
            "dataType": ["text"],
            "description": "type of entity to query. ex: headings",
            "name": "entityType",
        },
        {
            "dataType": ["DocInfo"],
            "description": "mapping of each query with a document",
            "name": "inDocs",
        },
        
    ] 
    },
    {
    "class": "DocInfo",
    "description": "Store page id and results",
    "properties":[
        {
            "dataType": ["int"],
            "description": "id of each doc",
            "name": "doc_id",
        },
        {
            "dataType": ["text"],
            "description": "path of the document",
            "name": "doc_path",
        },
        {
            "dataType": ["PageInfo"],
            "description": "mapping of each doc to page",
            "name": "hasPage",
        }
    ]
  },
    {
    "class": "PageInfo",
    "description": "Store page id and results",
    "properties":[
        {
            "dataType": ["ResultInfo"],
            "description": "results of each page",
            "name": "hasResult",
        },
        {
            "dataType": ["int"],
            "description": "id of each page",
            "name": "page_id",
        }
    ]
  },
     
   {
    "class": "ResultInfo",
    "description": "store answer text and bbox",
    "properties":[
        {
            "dataType": ["text"],
            "description": "Text of each word result",
            "name": "answer_text",
        },
        {
            "dataType": ["number[]"],
            "description": "Bbox of result",
            "name": "answer_bbox",
        }
    ]
  }
]
client.schema.create({"classes": schema})



def _createResultPageMap(page_data, weav_page_id):
    for res in page_data:
        weav_result_id = client.data_object.create(
                            data_object={"answer_text": res['text'],"answer_bbox": res['bbox'] },
                            class_name= "ResultInfo",
                        )
        
        client.data_object.reference.add(
            from_uuid = weav_page_id,
            from_property_name = "hasResult",
            to_uuid = weav_result_id
        )
        
        

        
def _create_docPageMap(page_results, weav_doc_id):
    for page_data in page_results:
          
        weav_page_id = client.data_object.create(
                    data_object = {"page_id": page_data['page_id']},
                    class_name = "PageInfo"
        )
        
        _createResultPageMap(page_data['page_answers'], weav_page_id)

        client.data_object.reference.add(
            from_uuid = weav_doc_id,
            from_property_name = "hasPage",
            to_uuid = weav_page_id
        )


def _create_resultObjs_add_db(weav_query_id, entity_results):
    for doc_data in entity_results:
        
        weav_doc_id =  client.data_object.create(
                    data_object = {"doc_id": doc_data['doc_id'], "doc_path": doc_data['doc_path']},
                    class_name = "DocInfo"
        )
        
        client.data_object.reference.add(
            from_uuid = weav_query_id,
            from_property_name = "inDocs",
            to_uuid = weav_doc_id
        )
        
        _create_docPageMap(doc_data['doc_results'], weav_doc_id)
    
    

def create_QueryEntityMap(entity_type, entity_info):
    weav_query_id =  client.data_object.create(
                    data_object = {"entity_type": entity_type},
                    class_name = "QueryEntityInfo",
                    vector = entity_info['vector']
        ) 
   
    _create_resultObjs_add_db(weav_query_id, entity_info['answers'] )

def createDBService(result):
    
    for key, value in result['query_info'].items(): 
        
        create_QueryEntityMap(key, value)


#result_info is the result from preprocessing service
createDBService(result_info)
 