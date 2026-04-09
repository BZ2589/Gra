import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.longlong)

    @staticmethod
    def _safe_divide(numerator, denominator):
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def Pixel_Accuracy(self):
        total = self.confusion_matrix.sum()
        Acc = self._safe_divide(np.diag(self.confusion_matrix).sum(), total)
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-7)
        mAcc = np.nanmean(Acc)
        return mAcc, Acc

    def Pixel_Precision_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        denominator = self.confusion_matrix[0, 1] + self.confusion_matrix[1, 1]
        Pre = self._safe_divide(self.confusion_matrix[1, 1], denominator)
        return Pre

    def Pixel_Recall_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        denominator = self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1]
        Rec = self._safe_divide(self.confusion_matrix[1, 1], denominator)
        return Rec

    def Pixel_F1_score(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.Pixel_Recall_Rate()
        Pre = self.Pixel_Precision_Rate()
        F1 = self._safe_divide(2 * Rec * Pre, Rec + Pre)
        return F1


    def calculate_per_class_metrics(self):
        # Adjustments to exclude class 0 in calculations
        TPs = np.diag(self.confusion_matrix)[1:]  # Start from index 1 to exclude class 0
        FNs = np.sum(self.confusion_matrix, axis=1)[1:] - TPs
        FPs = np.sum(self.confusion_matrix, axis=0)[1:] - TPs
        return TPs, FNs, FPs
    
    def Damage_F1_socore(self):
        TPs, FNs, FPs = self.calculate_per_class_metrics()
        precisions = TPs / (TPs + FPs + 1e-7)
        recalls = TPs / (TPs + FNs + 1e-7)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)
        return f1_scores
    
    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix) + 1e-7)
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Intersection_over_Union(self):
        denominator = self.confusion_matrix[0, 1] + self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1]
        IoU = self._safe_divide(self.confusion_matrix[1, 1], denominator)
        return IoU

    def Kappa_coefficient(self):
        num_total = np.sum(self.confusion_matrix)
        if num_total == 0:
            return 0.0

        observed_accuracy = self._safe_divide(np.trace(self.confusion_matrix), num_total)
        expected_accuracy = np.sum(
            np.sum(self.confusion_matrix, axis=0) / num_total * np.sum(self.confusion_matrix, axis=1) / num_total)

        return self._safe_divide(observed_accuracy - expected_accuracy, 1 - expected_accuracy)

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int64') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
