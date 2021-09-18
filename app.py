#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021-09-18 4:52 p.m.
# @Author : 
# @Version : 
# @File : app.py
# @desc :

from re import sub
from json import dumps
from flask import Flask, Response
from main import analyze

app = Flask(__name__)

@app.route("/<sentence>")
def index(sentence: str):
    return Response(
        dumps(list(analyze(sub('[^a-zA-Z ]', '', sentence)))),
        mimetype='application/json'
    )