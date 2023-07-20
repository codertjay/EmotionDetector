# Generated by Django 4.2.3 on 2023-07-14 15:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('emotions', '0002_report_name'),
    ]

    operations = [
        migrations.AddField(
            model_name='report',
            name='status',
            field=models.CharField(choices=[('ACTIVE', 'ACTIVE'), ('PENDING', 'PENDING')], default='PENDING', max_length=250),
        ),
    ]
