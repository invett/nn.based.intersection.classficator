def main(args):
    # Build Model
    model = get_model_resnet(args.resnetmodel, args.num_classes, args.triplet)

    if torch.cuda.is_available() and args.use_gpu:
        model = model.cuda()

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr, momentum=args.momentum)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    elif args.optimizer == 'ASGD':
        optimizer = torch.optim.ASGD(model.parameters(), args.lr)
    elif args.optimizer == 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), args.lr)
    else:
        print('not supported optimizer \n')
        exit()

    # create dataset and dataloader
    if args.triplet:
        pass
    else:
        pass

    # train model
    train(args, model, optimizer, dataloader_train, dataloader_val)


def validation(args, model, criterion, dataloader_val):
    print('\nstart val!')

    loss_record = 0.0
    acc_record = 0.0
    labelRecord = np.array([], dtype=np.uint8)
    predRecord = np.array([], dtype=np.uint8)

    with torch.no_grad():
        model.eval()

        for sample in dataloader_val:
            if args.triplet:
                anchor = sample['anchor']
                positive = sample['positive']
                negative = sample['negative']
            else:
                data = sample['data']
                label = sample['label']

            if torch.cuda.is_available() and args.use_gpu:
                if args.triplet:
                    anchor = anchor.cuda()
                    positive = positive.cuda()
                    negative = negative.cuda()
                else:
                    data = data.cuda()
                    label = label.cuda()

            if args.triplet:
                out_anchor = model(anchor)
                out_positive = model(positive)
                out_negative = model(negative)
            else:
                output = model(data)

            if args.triplet:
                loss = criterion(out_anchor, out_positive, out_negative)
            else:
                loss = criterion(output, label)

            loss_record += loss.item()

            if not args.triplet:
                predict = torch.argmax(output, 1)
                label = label.cpu().numpy()
                predict = predict.cpu().numpy()

                labelRecord = np.append(labelRecord, label)
                predRecord = np.append(predRecord, predict)

                acc_record += accuracy_score(label, predict)
            else:
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                acc_record += cos(out_anchor, out_positive)

    # Calculate validation metrics
    loss_val_mean = loss_record / len(dataloader_val)
    print('loss for test/validation : %f' % loss_val_mean)
    acc = acc_record / len(dataloader_val)
    print('Accuracy for test/validation : %f\n' % acc)

    return acc, loss_val_mean


def train(args, model, optimizer, dataloader_train, dataloader_val):
    if not os.path.isdir(args.save_model_path):
        os.mkdir(args.save_model_path)

    # starting values
    acc_pre = 0.0
    loss_pre = np.inf

    # Build criterion
    if args.triplet:
        criterion = torch.nn.TripletMarginLoss(margin=args.margin)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model.zero_grad()
    model.train()

    for epoch in range(args.num_epochs):
        lr = optimizer.param_groups[0]['lr']
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = 0.0
        acc_record = 0.0

        for sample in dataloader_train:
            if args.triplet:
                anchor = sample['anchor']
                positive = sample['positive']
                negative = sample['negative']
            else:
                data = sample['data']
                label = sample['label']

            if torch.cuda.is_available() and args.use_gpu:
                if args.triplet:
                    anchor = anchor.cuda()
                    positive = positive.cuda()
                    negative = negative.cuda()
                else:
                    data = data.cuda()
                    label = label.cuda()

            if args.triplet:
                out_anchor = model(anchor)
                out_positive = model(positive)
                out_negative = model(negative)
            else:
                output = model(data)

            if args.triplet:
                loss = criterion(out_anchor, out_positive, out_negative)
            else:
                loss = criterion(output, label)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)

            loss_record += loss.item()

            if args.triplet:
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                acc_record += cos(out_anchor, out_positive)
            else:
                predict = torch.argmax(output, 1)
                label = label.cpu().numpy()
                predict = predict.cpu().numpy()

                acc_record += accuracy_score(label, predict)

        tq.close()

        # Calculate metrics
        loss_train_mean = loss_record / len(dataloader_train)
        acc_train = acc_record / len(dataloader_train)
        print('loss for train : %f' % loss_train_mean)
        print('acc for train : %f' % acc_train)

        if epoch % args.validation_step == 0:

            acc_val, loss_val = validation(args, model, criterion, dataloader_val)

            if acc_pre < acc_val or loss_pre > loss_val:
                patience = 0
                if acc_pre < acc_val:
                    acc_pre = acc_val
                else:
                    loss_pre = loss_val

                bestModel = model.state_dict()
                if not args.triplet:
                    print('Best global accuracy: {}'.format(acc_pre))
                else:
                    print('Best global loss: {}'.format(loss_pre))
                print('Saving model: ', os.path.join(args.save_model_path, 'model_{}.pth'.format(args.resnetmodel)))
                torch.save(bestModel,
                           os.path.join(args.save_model_path, 'teacher_model_{}.pth'.format(args.resnetmodel)))

            elif epoch < args.patience_start:
                patience = 0

            else:
                patience += 1
                print('Patience: {}\n'.format(patience))

        if patience >= args.patience > 0:
            break
