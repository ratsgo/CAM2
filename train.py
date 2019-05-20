import sys, math
import tensorflow as tf
from config import Config
import lib.train_utils as train_utils
from lib.preprocess_utils import preprocessor, chat_to_ids


def train():

    config = Config()
    with tf.Session() as sess:

        is_training = True
        input_set, target_set = train_utils.read_train_set(config.id_path)
        batch = train_utils.get_batch(input_set,
                                      target_set,
                                      config.batch_size,
                                      config.max_epoch,
                                      config.sequence_length)

        model, _ = train_utils.create_or_load_model(sess, config)
        output_graph_def = sess.graph.as_graph_def()
        tf.train.write_graph(output_graph_def, config.checkpoint_path, config.graph_filename, as_text=True)
        checkpoint_loss = 0.0
        current_step = 0
        perplexity = 9999.0
        previous_loss = []

        while current_step < config.max_epoch and is_training and perplexity > config.perplexity_cut:

            input, output = next(batch)
            loss = model.step(sess, config, input, output, is_training)

            checkpoint_loss += loss / config.checkpoint_every
            current_step += 1

            if current_step % config.checkpoint_every == 0:
                perplexity = math.exp(checkpoint_loss)
                print("global step %d learning rate %.8f perplexity %.4f loss %.4f" %
                      (model.global_step.eval(), model.lr.eval(), perplexity, checkpoint_loss))

                if len(previous_loss) > 2 and checkpoint_loss > max(previous_loss[-2:]):
                    sess.run(model.lr_decay)
                previous_loss.append(checkpoint_loss)

                checkpoint_path = config.checkpoint_path + "model.ckpt"
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                checkpoint_loss = 0.0

                sys.stdout.flush()


def preprocess():
    config = Config()
    preprocessor(config)
    chat_to_ids(config)


if __name__ == '__main__':
    util_mode = sys.argv[1]
    if util_mode == "preprocess":
        preprocess()
    elif util_mode == "train":
        train()
    else:
        print("argument is wrong!")