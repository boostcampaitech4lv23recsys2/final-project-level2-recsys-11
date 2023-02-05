DROP TABLE IF EXISTS Experiments, Datasets;


create table Datasets (
    ID VARCHAR(20) NOT NULL,
    dataset_name VARCHAR(20) NOT NULL,
    dataset_desc VARCHAR(100) NOT NULL,
    upload_time VARCHAR(30) NOT NULL,
    train_interaction VARCHAR(255) NOT NULL,
    ground_truth VARCHAR(255) NOT NULL,
    user_side VARCHAR(255) NOT NULL,
    item_side VARCHAR(255) NOT NULL,

    PRIMARY KEY(ID, dataset_name),
    FOREIGN KEY(ID) REFERENCES Users(ID) ON DELETE CASCADE
);


CREATE TABLE Experiments (
    exp_id INT NOT NULL AUTO_INCREMENT,

    ID VARCHAR(20) NOT NULL, 
    dataset_name VARCHAR(20) NOT NULL,
    experiment_name VARCHAR(255) NOT NULL,
    alpha FLOAT NOT NULL,
    objective_fn VARCHAR(30),

    hyperparameters VARCHAR(255) NOT NULL,
    pred_items VARCHAR(255) NOT NULL,
    pred_scores VARCHAR(255),

    -- cosine_distance VARCHAR(100),
    -- pmi_distance VARCHAR(100),
    -- jaccard_distance VARCHAR(100),

    user_tsne VARCHAR(255),
    item_tsne VARCHAR(255),

    recall FLOAT NOT NULL,
    map FLOAT NOT NULL,
    ndcg FLOAT NOT NULL,
    tail_percentage FLOAT NOT NULL,
    avg_popularity FLOAT NOT NULL,
    coverage float NOT NULL,

    diversity_cos FLOAT NOT NULL,
    diversity_jac FLOAT,
    serendipity_pmi FLOAT NOT NULL,
    serendipity_jac FLOAT,
    novelty FLOAT NOT NULL,

    metric_per_user VARCHAR(255) NOT NULL,

    PRIMARY KEY (exp_id),
    FOREIGN KEY (ID, dataset_name) REFERENCES Datasets(ID, dataset_name) ON DELETE CASCADE

);