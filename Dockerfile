FROM debian:bullseye-slim

ENV INSTALL_DIR /flarestack

RUN apt-get update && apt-get dist-upgrade -y
RUN apt-get install python3 python3-pip -y

RUN python3 -m pip install poetry

ADD . ${INSTALL_DIR}

RUN cd ${INSTALL_DIR} && poetry install

# this still fails due  to lack fo sphinx-build executable
# RUN cd ${INSTALL_DIR} && poetry install --with docs
# RUN cd ${INSTALL_DIR} && sphinx-build -b html docs/source/ docs/build/html

# 'sh -c' allows variable expansion
CMD ["sh", "-c", "cd ${INSTALL_DIR} && poetry run python3 -m unittest discover"]
