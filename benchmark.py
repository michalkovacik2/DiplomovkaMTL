from HydraNet import *
from UTKFace import *
import torch
import argparse
import time


def load_annotations():
    train = []
    val = []
    with open('train_split.txt', 'r') as train_file:
        for line in train_file:
            train.append(line.strip())

    with open('val_split.txt', 'r') as val_file:
        for line in val_file:
            val.append(line.strip())

    return train, val


def init_parser(train):
    parser = argparse.ArgumentParser(prog='MultiTask learning')
    parser.add_argument('--batchSize',     type=int, help='batch size')
    parser.add_argument('--inputSize',     type=int, help='resolution of input')
    parser.add_argument('--backbone',      type=str, help='Type of backbone [resnet18, resnet101]')
    parser.add_argument('--ageHead',       type=int, help='If age head is present')
    parser.add_argument('--genderHead',    type=int, help='If gender head is present')
    parser.add_argument('--ethnicityHead', type=int, help='If ethnicity head is present')
    parser.add_argument('--model',         type=str, help='path to model', default=None)
    if train:
        parser.add_argument('--saveDir',    type=str, help='save directory')
        parser.add_argument('--endEpoch',   type=int, help='end of epoch')
        parser.add_argument('--checkpoint', type=int, help='Checkpoint interval when the model is saved')
        parser.add_argument('--stopWhen',   type=int,
                            help='Stop after the validation loss was not better after %in%', default=20)

    return parser


def print_parameters(args, train):
    print('Training init ...' if train else 'Get results init ...')
    print('Current parameters:')
    print(f'\tBatch size          = {args.batchSize}\n'
          f'\tModel path          = {args.model}\n'
          f'\tInput res.          = {args.inputSize} x {args.inputSize}\n'
          f'\tbackbone            = {args.backbone}\n'
          f'\tAge head            = {True if args.ageHead == 1 else False}\n'
          f'\tGender head         = {True if args.genderHead == 1 else False}\n'
          f'\tEthnicity head      = {True if args.ethnicityHead == 1 else False}')
    # Additional parameters for training
    if train:
        print(f'\tend epoch           = {args.endEpoch}\n'
              f'\tsave dir            = {args.saveDir}\n'
              f'\tcheckpoint interval = {args.checkpoint}\n'
              f'\tstop when           = {args.stopWhen}')


def main():
    parser = init_parser(train=False)
    args = parser.parse_args()

    head_for_age = True if args.ageHead == 1 else False
    head_for_gender = True if args.genderHead == 1 else False
    head_for_ethnicity = True if args.ethnicityHead == 1 else False

    train, val = load_annotations()

    val_dataloader = DataLoader(UTKFace(args.inputSize, val), shuffle=False, batch_size=args.batchSize)
    print(f'Train samples: {len(train)}, Validation samples: {len(val)}')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    model = HydraNet(args.backbone,
                     head_age=head_for_age,
                     head_gender=head_for_gender,
                     head_ethnicity=head_for_ethnicity).to(device=device)

    print_parameters(args, train=False)

    saved_model = torch.load(args.model)
    model.load_state_dict(saved_model['model_state_dict'])

    sig = nn.Sigmoid()

    with torch.no_grad():
        model.eval()

        TP_gender, FP_gender = 0, 0
        TP_race, FP_race = 0, 0
        total_error = 0

        total_time_start = time.time()
        for i, data in enumerate(val_dataloader):
            inputs = data['image'].to(device=device)

            age_label = data["age"].to(device=device)
            gender_label = data["gender"].to(device=device)
            race_label = data["ethnicity"].to(device=device)

            age_output, gender_output, race_output = model(inputs)

            if gender_output is not None:
                gender_res = torch.round(sig(gender_output))
                for p, l in zip(gender_res, gender_label):
                    if p.item() == l:
                        TP_gender += 1
                    else:
                        FP_gender += 1

            if race_output is not None:
                for p, l in zip(race_output, race_label):
                    race_res = torch.argmax(torch.nn.functional.softmax(p, dim=0)).item()
                    if race_res == l:
                        TP_race += 1
                    else:
                        FP_race += 1

            if age_output is not None:
                for p, l in zip(age_output, age_label):
                    total_error += abs(p - l)

        total_time = time.time() - total_time_start
        print(f'Total time: {total_time / 1000.0} ms')
        print(f'Avg. time per picture: {total_time / len(val) / 1000.0} ms')
        print(f'FPS: {len(val) / total_time}')
        print(f'Race:   TP = {TP_race}, FP = {FP_race}, % = {TP_race / len(val)}')
        print(f'Gender: TP = {TP_gender}, FP = {FP_gender}, % = {TP_gender / len(val)}')

        if type(total_error) is not int:
            print(f'Age: Mean avg. err = {total_error.item() / len(val)}')


if __name__ == '__main__':
    main()
