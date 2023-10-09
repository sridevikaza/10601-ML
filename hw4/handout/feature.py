import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map


def make_feature(glove_map, label, review):
    """
    Converts reviews to feature vectors by averaging GloVe embeddings.

    Parameters:
        glove_map (dict): dictionary of word embeddings

    Returns:
        np array feature vector for the given review
    """
    word_count = 0
    embedding_sum = np.zeros(300)
    review_words = review.split()
    for word in review_words:
        if word in glove_map:
            word_count += 1
            embedding_sum += glove_map[word]
    return np.insert(embedding_sum / word_count, 0, label)


def output(glove_file, in_file, out_file):
    """
    Reads input, converts x values to features, saves to output.

    Parameters:
        glove_file (str): File path to the glove embedding file.
        in_file (str): File path to read input.
        out_file (str): File path to save output.
    """
    glove_map = load_feature_dictionary(glove_file)
    input_data = load_tsv_dataset(in_file)
    output_arr = np.empty((0, 301))
    for row in input_data:
        feature = make_feature(glove_map, row[0], row[1])
        output_arr = np.vstack((output_arr, feature))
    np.savetxt(out_file, output_arr, delimiter='\t', fmt="%.6f")



if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()


    output(args.feature_dictionary_in, args.train_input, args.train_out)
    output(args.feature_dictionary_in, args.validation_input, args.validation_out)
    output(args.feature_dictionary_in, args.test_input, args.test_out)


# smalldata/train_small.tsv smalldata/val_small.tsv smalldata/test_small.tsv glove_embeddings.txt train_out.tsv val_out.tsv test_out.tsv