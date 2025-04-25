import traceback
from weaviate.util import get_valid_uuid # extract UUID from URL (beacon or href)




def page_result_map(page_info, map_name):
    result_beacons =  client.data_object.get(page_info['id'], with_vector=False)['properties'][map_name]
    for _beacon in result_beacons:
        result_uuid = get_valid_uuid(_beacon['beacon']) # can be 'href' too
        result_info = client.data_object.get(result_uuid, with_vector=False)
        print(result_info)
    
def doc_page_map(doc_val, map_name):
    page_beacons = client.data_object.get(doc_val['id'], with_vector=False)['properties'][map_name]
    for _beacon in page_beacons:
        page_uuid = get_valid_uuid(_beacon['beacon']) # can be 'href' too
        page_info = client.data_object.get(page_uuid, with_vector=False)
        
        page_result_map(page_info, "hasResult")
        
def _get_doc_info(query_ids):
    for doc_info in query_ids:
        doc_uuid = get_valid_uuid(doc_info['beacon']) # can be 'href' too
        doc_val = client.data_object.get(doc_uuid, with_vector=False)
        doc_page_map(doc_val, "hasPage")
        
        
def query_docs_map(class_name, map_name):
    query_objs = client.data_object.get(class_name=class_name)
    try:
        for each_query in query_objs['objects']:
            query_ids = client.data_object.get(each_query['id'], with_vector=False)['properties'][map_name]
            _get_doc_info(query_ids)
    except:
        print(traceback.print_exc())
        return None
    
#use query id to get the results 

query_docs_map(class_name =  "QueryEntityInfo", map_name = "inDocs")
