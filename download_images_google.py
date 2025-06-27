#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 10:09:03 2025

@author: kennyaskelson
"""


from icrawler.builtin import GoogleImageCrawler

def download_feathers(output_dir="Great_Horned_preprocessing", total_images=20):
    queries = ["Feather of Great Horned Owl"]
    #images_per_query = total_images // len(queries)

    crawler = GoogleImageCrawler(storage={"root_dir": output_dir})

    for query in queries:
        print(f"Downloading images for query: '{query}'")
        crawler.crawl(keyword=query, max_num=20)

if __name__ == "__main__":
    download_feathers()