#!/usr/bin/env bash

DB_HOST="localhost"
DB_PORT=""
DB_NAME=""
DB_USERNAME=""
DB_PASSWORD=""

# Database strings
CONNECTION_STRING="postgresql://${DB_USERNAME}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
OGR_CONNECTION_STRING="dbname='${DB_NAME}' user='${DB_USERNAME}' password='${DB_PASSWORD}' port='${DB_PORT}'"