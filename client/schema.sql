create table if not exists blobs (
  id int NOT NULL AUTO_INCREMENT,
  val MEDIUMBLOB NOT NULL,
  PRIMARY KEY (id)
);
create unique index id_idx on blobs (id) using BTREE;

create table if not exists config (
  max_id_on_ftp int
);

create table if not exists kv_storage (
  key_md5 VARCHAR(32) NOT NULL,
  k int NOT NULL,
  v int NOT NULL
);

create unique index k_idx on kv_storage (k) using BTREE;
create index key_md5_idx on kv_storage (key_md5) using BTREE;