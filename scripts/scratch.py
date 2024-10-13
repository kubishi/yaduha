import json
import pathlib
from typing import Dict, List, Union

thisdir = pathlib.Path(__file__).parent.resolve()

def main():
    path_new = thisdir / "results" / "evaluation_results.json"
    path_old = thisdir / "results" / "evaluation_results_old.json"

    new: Dict[str, List[Dict[str, Union[Dict,str,float,int]]]] = json.loads(path_new.read_text())
    old: Dict[str, List[Dict[str, Union[Dict,str,float,int]]]] = json.loads(path_old.read_text())
    metadata = {
        (det['translator'], det['model'], det['translation']['source']): det
        for det in old['results']
    }
    print(metadata)
    combined = {'results': []}
    for det in new['results']:
        key = (det['translator'], det['model'], det['translation']['source'])
        print(key)
        if key in metadata:
            print('here')
            det.update(metadata[key])

        combined['results'].append(det)

    path_new.write_text(json.dumps(combined, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

