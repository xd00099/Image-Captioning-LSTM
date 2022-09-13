################################################################################
# CSE 151B: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin
# Winter 2022
################################################################################

from email.mime import base
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class baselineLSTM(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab):
        super(baselineLSTM, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet50.children())[:-1])
        self.fc = nn.Linear(resnet50.fc.in_features, embedding_size)
        self.embed = nn.Embedding(len(vocab), embedding_size)
        self.decoder = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers = 2, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, len(vocab))
    
    def forward(self, images, captions):
        with torch.no_grad():
            features = self.encoder(images)

        features = features.reshape(features.size(0), -1)
        features = self.fc(features)
        embeddings = self.embed(captions[:,:-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        out, _ = self.decoder(embeddings)
        out = self.fc_out(out)
        return out

    def sample(self, images, vocab, max_length, deterministic, temp):
        
        images = images.cuda()
        with torch.no_grad():
            features = self.encoder(images)
        
        # first time stamp
        features = features.reshape(features.size(0), -1)
        features = self.fc(features.unsqueeze(1))
        out, h_state = self.decoder(features)
        out = self.fc_out(out.squeeze(1))
        sampled_results = []

        if deterministic:
            pred = out.argmax(1)
            sampled_results.append(pred)
        else:
            probs = F.softmax(torch.div(out, temp), 1)
            pred = torch.multinomial(probs, 1).squeeze(1)
            sampled_results.append(pred)


        # embedding layers
        for _ in range(1, max_length):
            out = self.embed(pred).unsqueeze(1)
            r_out, h_state = self.decoder(out, h_state)
            out = self.fc_out(r_out.squeeze(1))

            if deterministic:
                pred = out.argmax(1)
                sampled_results.append(pred)
            else:
                probs = F.softmax(torch.div(out, temp), 1)
                pred = torch.multinomial(probs, 1).squeeze(1)
                sampled_results.append(pred)


        # get actual words
        sampled_captions = []
        for i in range(len(sampled_results[0])):
            pred_captions = []
            for j in range(max_length):
                pred_word = vocab.idx2word[sampled_results[j][i].item()]
                if pred_word == '<end>':
                    break
                elif pred_word == '<start>':
                    continue

                pred_captions.append(pred_word)
            sampled_captions.append(pred_captions)
        
        return sampled_captions

class baselineRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab):
        super(baselineRNN, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet50.children())[:-1])
        self.fc = nn.Linear(resnet50.fc.in_features, embedding_size)
        self.embed = nn.Embedding(len(vocab), embedding_size)
        self.decoder = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers = 2, nonlinearity='relu', batch_first=True)
        self.fc_out = nn.Linear(hidden_size, len(vocab))
    
    def forward(self, images, captions):
        with torch.no_grad():
            features = self.encoder(images)

        features = features.reshape(features.size(0), -1)
        features = self.fc(features)
        embeddings = self.embed(captions[:,:-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        out, _ = self.decoder(embeddings)
        out = self.fc_out(out)
        return out

    def sample(self, images, vocab, max_length, deterministic, temp):
        
        images = images.cuda()
        with torch.no_grad():
            features = self.encoder(images)
        
        # first time stamp
        features = features.reshape(features.size(0), -1)
        features = self.fc(features.unsqueeze(1))
        out, h_state = self.decoder(features)
        out = self.fc_out(out.squeeze(1))
        sampled_results = []

        if deterministic:
            pred = out.argmax(1)
            sampled_results.append(pred)
        else:
            probs = F.softmax(torch.div(out, temp), 1)
            pred = torch.multinomial(probs, 1).squeeze(1)
            sampled_results.append(pred)


        # embedding layers
        for _ in range(1, max_length):
            out = self.embed(pred).unsqueeze(1)
            r_out, h_state = self.decoder(out, h_state)
            out = self.fc_out(r_out.squeeze(1))

            if deterministic:
                pred = out.argmax(1)
                sampled_results.append(pred)
            else:
                probs = F.softmax(torch.div(out, temp), 1)
                pred = torch.multinomial(probs, 1).squeeze(1)
                sampled_results.append(pred)


        # get actual words
        sampled_captions = []
        for i in range(len(sampled_results[0])):
            pred_captions = []
            for j in range(max_length):
                pred_word = vocab.idx2word[sampled_results[j][i].item()]
                if pred_word == '<end>':
                    break
                elif pred_word == '<start>':
                    continue

                pred_captions.append(pred_word)
            sampled_captions.append(pred_captions)
        
        return sampled_captions

class LSTM2(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab):
        super(LSTM2, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.embedding_size = embedding_size
        self.encoder = nn.Sequential(*list(resnet50.children())[:-1])
        self.fc = nn.Linear(resnet50.fc.in_features, embedding_size)
        self.embed = nn.Embedding(len(vocab), embedding_size)
        self.decoder = nn.LSTM(input_size=2*embedding_size, hidden_size=hidden_size, num_layers = 2, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, len(vocab))
    
    def forward(self, images, captions):
        with torch.no_grad():
            features = self.encoder(images)
        
        pad = torch.zeros(images.shape[0],1, dtype = torch.long).cuda()
        captions = torch.cat((pad, captions), dim=1)
        features = features.reshape(features.size(0), -1)
        features = self.fc(features)
        features = features.unsqueeze(1).repeat((1,captions.shape[1]-1,1))
        embeddings = self.embed(captions[:,:-1])

        embeddings = torch.cat((embeddings, features), 2)
        out, _ = self.decoder(embeddings)
        out = self.fc_out(out)
        return out

    def sample(self, images, vocab, max_length, deterministic, temp):
        
        images = images.cuda()
        with torch.no_grad():
            features = self.encoder(images)
        
        # first time stamp
        pad = torch.zeros(images.shape[0],1, dtype=torch.long).cuda()
        pad_embed = self.embed(pad)
        features = features.reshape(features.size(0), -1)
        features = self.fc(features.unsqueeze(1))
        init_features = torch.cat((pad_embed, features), 2)
        out, h_state = self.decoder(init_features)
        out = self.fc_out(out.squeeze(1))
        sampled_results = []

        if deterministic:
            pred = out.argmax(1)
            sampled_results.append(pred)
        else:
            probs = F.softmax(torch.div(out, temp), 1)
            pred = torch.multinomial(probs, 1).squeeze(1)
            sampled_results.append(pred)


        # embedding layers
        for _ in range(1, max_length):
            
            out = self.embed(pred).unsqueeze(1)
            out_features = torch.cat((out, features), 2)
            r_out, h_state = self.decoder(out_features, h_state)
            out = self.fc_out(r_out.squeeze(1))

            if deterministic:
                pred = out.argmax(1)
                sampled_results.append(pred)
            else:
                probs = F.softmax(torch.div(out, temp), 1)
                pred = torch.multinomial(probs, 1).squeeze(1)
                sampled_results.append(pred)


        # get actual words
        sampled_captions = []
        for i in range(len(sampled_results[0])):
            pred_captions = []
            for j in range(max_length):
                pred_word = vocab.idx2word[sampled_results[j][i].item()]
                if pred_word == '<end>':
                    break
                elif pred_word == '<start>':
                    continue

                pred_captions.append(pred_word)
            sampled_captions.append(pred_captions)
        
        return sampled_captions
    
    
# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    # You may add more parameters if you want
    if model_type == 'LSTM':
        model = baselineLSTM(hidden_size, embedding_size, vocab)
    elif model_type == 'RNN':
        model = baselineRNN(hidden_size, embedding_size, vocab)
    elif model_type == 'LSTM2':
        model = LSTM2(hidden_size, embedding_size, vocab)
    return model