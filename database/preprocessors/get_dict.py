import sys
from lxml import etree
import pickle as pkl
from rdkit import Chem


def get_smiles_dict(root):
    smiles_dict = {} # key:drugbank_id, value:smiles

    for drug in root.xpath('./*[local-name()="drug"]'):
        drug_id = drug.xpath('./*[local-name()="drugbank-id"][@primary="true"]')[0].text
        
        for kind in drug.xpath('.//*[local-name()="kind"]'):
            if kind.text == 'SMILES':
                smiles = kind.getparent()[1].text
                # Confirm the validity of smiles
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    n_atoms = int(mol.GetNumAtoms())
                    if n_atoms > 200:
                        break
                except:
                    break

                smiles_dict[drug_id] = smiles
    return smiles_dict

def get_name_dict(root, smiles_dict):
    name_dict = {} # key:name, value:drugbank_id

    for drug in root.xpath('./*[local-name()="drug"]'):
        drug_id = drug.xpath('./*[local-name()="drugbank-id"][@primary="true"]')[0].text

        if drug_id not in smiles_dict:
            continue

        # Name
        name_text = drug.xpath('./*[local-name()="name"]')[0].text
        name_dict[name_text] = drug_id
        # Brand
        for brand in drug.xpath('./*[local-name()="international-brands"]')[0]:
            brand_text = brand.xpath('./*[local-name()="name"]')[0].text
            name_dict[brand_text] = drug_id
        # Product
        for product in drug.xpath('./*[local-name()="products"]')[0]:
            product_text = product.xpath('./*[local-name()="name"]')[0].text
            name_dict[product_text] = drug_id
        # Pfam
        pfams = drug.xpath('.//*[local-name()="pfams"]')
        if len(pfams) != 0:
            for pfam in pfams[0]:
                pfam_text = pfam.xpath('./*[local-name()="name"]')[0].text
                name_dict[pfam_text] = drug_id
    return name_dict

if len(sys.argv) != 4:
    sys.stderr.write('Usage: python3 %s drugbank_xml name_dict_path smiles_dict_path' % (sys.argv[0]))
    sys.exit(-1)

root = etree.parse(sys.argv[1], parser=etree.XMLParser())
smiles_dict = get_smiles_dict(root)
name_dict = get_name_dict(root, smiles_dict)

with open(sys.argv[2], 'wb') as f:
    pkl.dump(name_dict, f)
with open(sys.argv[3], 'wb') as f:
    pkl.dump(smiles_dict, f)
