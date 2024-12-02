import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path

import pandas as pd
from guided_diffusion import filename_utils

def make_link(src_fp,trg_fp):
    if os.path.islink(trg_fp):
        os.remove(trg_fp)
    os.symlink(src_fp,trg_fp)

site_map = {
    "BLCA":"Bladder",
    "BRCA":"Breast",
    "COAD":"Colorectal",
    "ESCA":"Esphagogastric",
    "HNSC":"Head Neck",
    "KICH":"Renal",
    "KIRC":"Renal",
    "KIRP":"Renal",
    "LIHC":"Liver",
    "LUAD":"Lung",
    "LUSC":"Lung",
    "OV":"Ovarian",
    "PAAD":"Pancreatic",
    "PRAD":"Prostate",
    "READ":"Colorectal",
    "SARC":"Sarcoma",
    "STAD":"Esphagogastric",
    "THCA":"Thyroid",
    "UCEC":"Endometrial"
}

def map_tumor_site(project_id):
    return site_map[project_id.split("-")[1]]

def main1():
    trans_dir="/encrypted/SFBData/liz0f/guided-diffusion/checkpoints/omics_TCGA_newembed-single_mod-128x128-ddib-cmd/translation"
    search_dir = "mutation_images/image_search_examples2"
    fp_info_df=filename_utils.parse_directory_filenames(trans_dir)
    query_fn_list = fp_info_df.query("(slide_class=='TCGA-COAD')&(patch_type=='processed')")["fn"].values.tolist()
    database_fn_list = fp_info_df.query("(slide_class=='TCGA-COAD')&(patch_type=='translated')")["fn"].values.tolist()
    
    os.makedirs(os.path.join(search_dir,"database_images"),exist_ok=True)
    os.makedirs(os.path.join(search_dir,"query_images"),exist_ok=True)

    for fn in query_fn_list:
        newfn = "__".join(fn.split("__")[1:]).replace("_","__",1)
        if os.path.islink(os.path.join(search_dir,"query_images",newfn)):
            os.remove(os.path.join(search_dir,"query_images",newfn))
        os.symlink(
            os.path.join(trans_dir,fn),
            os.path.join(search_dir,"query_images",newfn)
        )

    for fn in database_fn_list:
        newfn = "__".join(fn.split("__")[1:]).replace("_","__",1)
        if os.path.islink(os.path.join(search_dir,"database_images",newfn)):
            os.remove(os.path.join(search_dir,"database_images",newfn))

        os.symlink(
            os.path.join(trans_dir,fn),
            os.path.join(search_dir,"database_images",newfn)
        )

def main2():
    trans_dir="/encrypted/SFBData/liz0f/guided-diffusion/checkpoints/FFPE_frozen_TCGA-128x128-ddib-cmd/translation"
    search_dir = "mutation_images/frozen_translation_search"
    fp_info_df=filename_utils.parse_directory_filenames(trans_dir)
    diagnostic_slide_fn_list = fp_info_df.query("(patch_type=='processed')&(slide_class=='diagnostic_slide')")["fn"].values.tolist()
    tissue_slide_fn_list = fp_info_df.query("(patch_type=='processed')&(slide_class=='tissue_slide')")["fn"].values.tolist()
    translated_tissue_slide_fn_list = fp_info_df.query("(patch_type=='translated')&(slide_class=='tissue_slide')&(translated_slide_class=='diagnostic_slide')")["fn"].values.tolist()

    TCGA_info_fp="/encrypted/SFBData/liz0f/da_graph/dataset_files/TCGA/info/TCGA_19types.csv"
    TCGA_info=pd.read_csv(TCGA_info_fp,index_col=0)

    os.makedirs(os.path.join(search_dir,"database_images"),exist_ok=True)
    for fn in diagnostic_slide_fn_list:
        fn_info_dict = filename_utils.parse_patch_filename(fn)
        slide_id = fn_info_dict["slide_id"]
        site_class = map_tumor_site(TCGA_info.loc[slide_id,"project_id"])
        newfn = site_class+"__"+fn
        make_link(
            os.path.join(trans_dir,fn),
            os.path.join(search_dir,"database_images",newfn)
        )
    
    os.makedirs(os.path.join(search_dir,"diagnostic_slide_query_images"),exist_ok=True)
    for fn in diagnostic_slide_fn_list:
        fn_info_dict = filename_utils.parse_patch_filename(fn)
        slide_id = fn_info_dict["slide_id"]
        site_class = map_tumor_site(TCGA_info.loc[slide_id,"project_id"])
        newfn = site_class+"__"+fn
        make_link(
            os.path.join(trans_dir,fn),
            os.path.join(search_dir,"diagnostic_slide_query_images",newfn)
        )

    os.makedirs(os.path.join(search_dir,"tissue_slide_query_images"),exist_ok=True)
    for fn in tissue_slide_fn_list:
        fn_info_dict = filename_utils.parse_patch_filename(fn)
        slide_id = fn_info_dict["slide_id"]
        site_class = map_tumor_site(TCGA_info.loc[slide_id,"project_id"])
        newfn = site_class+"__"+fn
        make_link(
            os.path.join(trans_dir,fn),
            os.path.join(search_dir,"tissue_slide_query_images",newfn)
        )
    
    os.makedirs(os.path.join(search_dir,"translated_slide_query_images"),exist_ok=True)
    for fn in translated_tissue_slide_fn_list:
        fn_info_dict = filename_utils.parse_patch_filename(fn)
        slide_id = fn_info_dict["slide_id"]
        site_class = map_tumor_site(TCGA_info.loc[slide_id,"project_id"])
        newfn = site_class+"__"+fn
        make_link(
            os.path.join(trans_dir,fn),
            os.path.join(search_dir,"translated_slide_query_images",newfn)
        )
if __name__=="__main__":
    main1()
