import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useAuth } from '@/contexts/AuthContext';
import { api, VitalsSubmitResponse } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';
import { Thermometer } from 'lucide-react';

interface VitalsFormProps {
  onSubmitSuccess: (data: VitalsSubmitResponse) => void;
}

const VitalsForm = ({ onSubmitSuccess }: VitalsFormProps) => {
  const { token, accountType } = useAuth();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  
  const [vitals, setVitals] = useState({
    age: '',
    systolic_bp: '',
    diastolic_bp: '',
    bs: '',
    body_temp: '',
    body_temp_unit: 'celsius' as 'celsius' | 'fahrenheit',
    heart_rate: '',
    patient_history: '',
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!token || !accountType) {
      toast({
        title: 'Error',
        description: 'Please login again',
        variant: 'destructive',
      });
      return;
    }

    try {
      setIsLoading(true);
      const payload = {
        vitals: {
          age: Number(vitals.age),
          systolic_bp: Number(vitals.systolic_bp),
          diastolic_bp: Number(vitals.diastolic_bp),
          bs: Number(vitals.bs),
          body_temp: Number(vitals.body_temp),
          body_temp_unit: vitals.body_temp_unit,
          heart_rate: Number(vitals.heart_rate),
          patient_history: vitals.patient_history,
        },
        account_type: accountType as 'pregnant' | 'postnatal' | 'general',
      };

      const response = await api.submitVitals(payload, token);
      onSubmitSuccess(response);
      
      toast({
        title: 'Success',
        description: 'Vitals submitted successfully',
      });

      // Reset form
      setVitals({
        age: '',
        systolic_bp: '',
        diastolic_bp: '',
        bs: '',
        body_temp: '',
        body_temp_unit: 'celsius',
        heart_rate: '',
        patient_history: '',
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: error instanceof Error ? error.message : 'Failed to submit vitals',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Thermometer className="h-5 w-5" />
          Submit Your Vitals
        </CardTitle>
        <CardDescription>Enter your current vital signs for AI-powered risk assessment</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="age">Age (years)</Label>
              <Input
                id="age"
                type="number"
                value={vitals.age}
                onChange={(e) => setVitals({ ...vitals, age: e.target.value })}
                placeholder="e.g., 28"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="heart_rate">Heart Rate (bpm)</Label>
              <Input
                id="heart_rate"
                type="number"
                value={vitals.heart_rate}
                onChange={(e) => setVitals({ ...vitals, heart_rate: e.target.value })}
                placeholder="e.g., 75"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="systolic_bp">Systolic BP (mmHg)</Label>
              <Input
                id="systolic_bp"
                type="number"
                value={vitals.systolic_bp}
                onChange={(e) => setVitals({ ...vitals, systolic_bp: e.target.value })}
                placeholder="e.g., 120"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="diastolic_bp">Diastolic BP (mmHg)</Label>
              <Input
                id="diastolic_bp"
                type="number"
                value={vitals.diastolic_bp}
                onChange={(e) => setVitals({ ...vitals, diastolic_bp: e.target.value })}
                placeholder="e.g., 80"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="bs">Blood Sugar (mmol/L)</Label>
              <Input
                id="bs"
                type="number"
                step="0.1"
                value={vitals.bs}
                onChange={(e) => setVitals({ ...vitals, bs: e.target.value })}
                placeholder="e.g., 5.5"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="body_temp">Body Temperature</Label>
              <div className="flex gap-2">
                <Input
                  id="body_temp"
                  type="number"
                  step="0.1"
                  value={vitals.body_temp}
                  onChange={(e) => setVitals({ ...vitals, body_temp: e.target.value })}
                  placeholder="e.g., 37.0"
                  required
                  className="flex-1"
                />
                <Select 
                  value={vitals.body_temp_unit} 
                  onValueChange={(value: 'celsius' | 'fahrenheit') => 
                    setVitals({ ...vitals, body_temp_unit: value })
                  }
                >
                  <SelectTrigger className="w-28">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="celsius">°C</SelectItem>
                    <SelectItem value="fahrenheit">°F</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="patient_history">Patient History (Optional)</Label>
            <Textarea
              id="patient_history"
              value={vitals.patient_history}
              onChange={(e) => setVitals({ ...vitals, patient_history: e.target.value })}
              placeholder="Any relevant medical history or current symptoms..."
              rows={3}
            />
          </div>

          <Button type="submit" className="w-full" disabled={isLoading}>
            {isLoading ? 'Processing...' : 'Submit Vitals'}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
};

export default VitalsForm;
