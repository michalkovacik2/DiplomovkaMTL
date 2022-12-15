from HydraNet import *
import os
from sklearn.model_selection import train_test_split
from UTKFace import *
import torch
import glob
import cv2
import argparse
import math


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
    parser.add_argument('--batchSize',     type=int,   help='batch size')
    parser.add_argument('--inputSize',     type=int,   help='resolution of input')
    parser.add_argument('--backbone',      type=str,   help='Type of backbone [resnet18, resnet101]')
    parser.add_argument('--ageHead',       type=int,   help='If age head is present')
    parser.add_argument('--genderHead',    type=int,   help='If gender head is present')
    parser.add_argument('--ethnicityHead', type=int,   help='If ethnicity head is present')
    parser.add_argument('--model',         type=str,   help='path to model', default=None)
    if train:
        parser.add_argument('--ageCoef',       type=float, help='Coefficient for age head', default=1.0)
        parser.add_argument('--genderCoef',    type=float, help='Coefficient for gender head', default=1.0)
        parser.add_argument('--ethnicityCoef', type=float, help='Coefficient for ethnicity head', default=1.0)
        parser.add_argument('--saveDir',       type=str,  help='save directory')
        parser.add_argument('--endEpoch',      type=int,  help='end of epoch')
        parser.add_argument('--checkpoint',    type=int,  help='Checkpoint interval when the model is saved')
        parser.add_argument('--stopWhen',      type=int,
                            help='Stop after the validation loss was not better after %in%', default=20)

    return parser


def print_parameters(args, train):
    print('Training init ...' if train else 'Get results init ...')
    print('Current parameters:')
    print(f'\tBatch size            = {args.batchSize}\n'
          f'\tModel path            = {args.model}\n'
          f'\tInput res.            = {args.inputSize} x {args.inputSize}\n'
          f'\tbackbone              = {args.backbone}\n'
          f'\tAge head              = {True if args.ageHead == 1 else False}\n'
          f'\tGender head           = {True if args.genderHead == 1 else False}\n'
          f'\tEthnicity head        = {True if args.ethnicityHead == 1 else False}')
    # Additional parameters for training
    if train:
        print(f'\tend epoch             = {args.endEpoch}\n'
              f'\tsave dir              = {args.saveDir}\n'
              f'\tcheckpoint interval   = {args.checkpoint}\n'
              f'\tstop when             = {args.stopWhen}\n'
              f'\tage coefficient       = {args.ageCoef}\n'
              f'\tgender coefficient    = {args.genderCoef}\n'
              f'\tethnicity coefficient = {args.ethnicityCoef}')


def main():
    parser = init_parser(train=True)
    args = parser.parse_args()

    head_for_age = True if args.ageHead == 1 else False
    head_for_gender = True if args.genderHead == 1 else False
    head_for_ethnicity = True if args.ethnicityHead == 1 else False

    # If model is None try to load the last saved trained_model*.pth
    epoch_start = 0
    if args.model is None:
        filenames = glob.glob(f'{args.saveDir}trained_model*.pth')
        if len(filenames) != 0:
            filenames_num = [int(f.split(f'{args.saveDir}trained_model')[1].split('.pth')[0]) for f in filenames]
            idx = 0
            max_val = filenames_num[0]
            for idx_file in range(len(filenames_num)):
                if filenames_num[idx_file] > max_val:
                    max_val = filenames_num[idx_file]
                    idx = idx_file

            args.model = filenames[idx]
            epoch_start = max_val

    # Print current parameters
    print_parameters(args, train=True)

    # dataset = os.listdir('UTKFace/')
    # train, val = train_test_split(dataset, test_size=0.2, train_size=0.8, random_state=4)

    train, val = load_annotations()

    train_dataloader = DataLoader(UTKFace(args.inputSize, train), shuffle=True, batch_size=args.batchSize)
    val_dataloader = DataLoader(UTKFace(args.inputSize, val), shuffle=False, batch_size=args.batchSize)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print('Warning!!!: Training on cpu')

    model = HydraNet(args.backbone,
                     head_age=head_for_age,
                     head_gender=head_for_gender,
                     head_ethnicity=head_for_ethnicity).to(device=device)

    # Load old model if present
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.09)
    if args.model is not None:
        saved_model = torch.load(args.model)
        model.load_state_dict(saved_model['model_state_dict'])
        optimizer.load_state_dict(saved_model['optimzier_state_dict'])

    # define loss functions
    ethnicity_loss = nn.CrossEntropyLoss()
    gender_loss = nn.BCELoss()
    age_loss = nn.L1Loss()

    sig = nn.Sigmoid()

    best_loss = math.inf
    best_loss_epoch = -1

    for epoch in range(epoch_start + 1, args.endEpoch + 1):
        model.train(True)

        total_age_loss = 0
        total_gender_loss = 0
        total_race_loss = 0
        total_training_loss = 0

        for i, data in enumerate(train_dataloader):
            inputs = data['image'].to(device=device)

            age_label = data["age"].to(device=device)
            gender_label = data["gender"].to(device=device)
            race_label = data["ethnicity"].to(device=device)

            optimizer.zero_grad()

            age_output, gender_output, race_output = model(inputs)

            loss1 = 0 if race_output is None else ethnicity_loss(race_output, race_label)
            loss1 *= args.ethnicityCoef
            loss2 = 0 if gender_output is None else gender_loss(sig(gender_output), gender_label.unsqueeze(1).float())
            loss2 *= args.genderCoef
            loss3 = 0 if age_output is None else age_loss(age_output, age_label.unsqueeze(1).float())
            loss3 *= args.ageCoef
            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()

            total_training_loss += loss
            total_age_loss += loss3
            total_gender_loss += loss2
            total_race_loss += loss1

        print(f'{epoch},{total_training_loss:.4f},{total_age_loss:.4f},'
              f' {total_gender_loss:.4f},{total_race_loss:.4f}')

        if epoch % args.checkpoint == 0 and epoch != 0:
            # Test inference with validation set
            with torch.no_grad():
                model.eval()

                total_age_loss_val = 0
                total_gender_loss_val = 0
                total_race_loss_val = 0
                total_validation_loss_val = 0

                for i, data in enumerate(val_dataloader):
                    inputs = data['image'].to(device=device)

                    age_label = data["age"].to(device=device)
                    gender_label = data["gender"].to(device=device)
                    race_label = data["ethnicity"].to(device=device)

                    optimizer.zero_grad()

                    age_output, gender_output, race_output = model(inputs)

                    loss1 = 0 if race_output is None else ethnicity_loss(race_output, race_label)
                    loss1 *= args.ethnicityCoef
                    loss2 = 0 if gender_output is None else gender_loss(sig(gender_output),
                                                                        gender_label.unsqueeze(1).float())
                    loss2 *= args.genderCoef
                    loss3 = 0 if age_output is None else age_loss(age_output, age_label.unsqueeze(1).float())
                    loss3 *= args.ageCoef
                    loss = loss1 + loss2 + loss3

                    total_validation_loss_val += loss
                    total_age_loss_val += loss3
                    total_gender_loss_val += loss2
                    total_race_loss_val += loss1

                print(f'VAL: {epoch},{total_validation_loss_val:.4f},{total_age_loss_val:.4f}, '
                      f'{total_gender_loss_val:.4f},{total_race_loss_val:.4f}')

            if total_validation_loss_val < best_loss:
                print(f'Found better model loss = {total_validation_loss_val} epoch = {epoch}')
                best_loss = total_validation_loss_val
                best_loss_epoch = epoch
                torch.save({"model_state_dict": model.state_dict(),
                            "optimzier_state_dict": optimizer.state_dict()
                            }, f"{args.saveDir}best.pth")

            torch.save({"model_state_dict": model.state_dict(),
                        "optimzier_state_dict": optimizer.state_dict()
                        }, f"{args.saveDir}last.pth")

            if best_loss_epoch != -1 and (epoch - best_loss_epoch) >= args.stopWhen:
                print(f'We did not find better loss after {args.stopWhen} epochs. ENDING TRAINING')
                return


def show_results():
    SAVE_DIR = 'resnet101/'

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    dataset = os.listdir('UTKFace/')
    train, val = train_test_split(dataset, test_size=0.2, train_size=0.8, random_state=4)
    val_dataloader = DataLoader(UTKFace(val), shuffle=False, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = HydraNet('resnet18').to(device=device)

    # Load old model if present
    filenames = glob.glob(f'{SAVE_DIR}trained_model*Old.pth')
    filenames_num = [int(f.split(f'{SAVE_DIR}trained_model')[1].split('Old.pth')[0]) for f in filenames]
    idx = 0
    max_val = filenames_num[0]
    for i in range(len(filenames_num)):
        if filenames_num[i] > max_val:
            max_val = filenames_num[i]
            idx = i

    saved_model = torch.load(filenames[idx])
    model.load_state_dict(saved_model['model_state_dict'])
    sig = nn.Sigmoid()

    with torch.no_grad():
        model.eval()

        for i, data in enumerate(val_dataloader):
            inputs = data['image'].to(device=device)

            age_label = data["age"].to(device=device)
            gender_label = data["gender"].to(device=device)
            race_label = data["ethnicity"].to(device=device)

            age_output, gender_output, race_output = model(inputs)
            img = cv2.imread(f'UTKFace/{val[i]}')
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 0, 255)

            age_predction = int(age_output.squeeze(0).item())
            gender_prediction = int(torch.round(sig(gender_output)).item())
            race_prediction = torch.argmax(torch.nn.functional.softmax(race_output, dim=1)).item()

            gt = f'GT: {age_label.item()}, {gender_label.item()}, {race_label.item()}'
            dt = f'DT: {age_predction}, {gender_prediction}, {race_prediction}'
            img = cv2.putText(img, gt, (0, 15), font, 0.6, color, 1, cv2.LINE_AA)
            img = cv2.putText(img, dt, (0, 35), font, 0.6, color, 1, cv2.LINE_AA)
            cv2.imshow('img', img)
            c = cv2.waitKey(0)
            if c == ord('q'):
                return


def get_results():
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

    ethnicity_loss = nn.CrossEntropyLoss()
    gender_loss = nn.BCELoss()
    age_loss = nn.L1Loss()

    sig = nn.Sigmoid()

    with torch.no_grad():
        model.eval()

        total_age_loss_val = 0
        total_gender_loss_val = 0
        total_race_loss_val = 0
        total_training_loss_val = 0

        TP_gender, FP_gender = 0, 0
        TP_race, FP_race = 0, 0
        total_error = 0

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

            loss1 = 0 if race_output is None else ethnicity_loss(race_output, race_label)
            loss2 = 0 if gender_output is None else gender_loss(sig(gender_output),
                                                                gender_label.unsqueeze(1).float())
            loss3 = 0 if age_output is None else age_loss(age_output, age_label.unsqueeze(1).float())
            loss = loss1 + loss2 + loss3

            total_training_loss_val += loss
            total_age_loss_val += loss3
            total_gender_loss_val += loss2
            total_race_loss_val += loss1

        print(f'total loss: {total_training_loss_val:.4f}, age loss: {total_age_loss_val:.4f}, '
              f'gender loss: {total_gender_loss_val:.4f}, ethnicity: {total_race_loss_val:.4f}')
        print(f'Race:   TP = {TP_race}, FP = {FP_race}, % = {TP_race / len(val)}')
        print(f'Gender: TP = {TP_gender}, FP = {FP_gender}, % = {TP_gender / len(val)}')

        if type(total_error) is not int:
            print(f'Age: Mean avg. err = {total_error.item() / len(val)}')


if __name__ == '__main__':
    # show_results()
    main()
    # get_results()
