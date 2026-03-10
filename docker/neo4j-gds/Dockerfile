ARG NEO4J_VERSION=5.26.21-community
FROM neo4j:${NEO4J_VERSION}

ARG GDS_VERSION=2.13.7

USER root
ADD https://graphdatascience.ninja/neo4j-graph-data-science-${GDS_VERSION}.jar /plugins/neo4j-graph-data-science.jar
RUN chown neo4j:neo4j /plugins/neo4j-graph-data-science.jar
USER neo4j
